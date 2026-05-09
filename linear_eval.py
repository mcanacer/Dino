import os
import sys

import yaml

import jax
import jax.numpy as jnp
import numpy as np

import optax
import flax.linen as nn
from flax import serialization
import orbax.checkpoint as ocp

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import wandb

import vit


# ---------- checkpoints ----------------------------------------------------

def save_eval_state(path, state):
    """Save the linear classifier's own state (not the pretrained ViT)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_eval_state(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def load_pretrained_backbone_params(checkpoint_dir, source="teacher"):
    """Restore an Orbax training checkpoint and return only the backbone
    params, wrapped as a Flax variables dict so that ``backbone.apply`` works.

    ``source`` selects 'teacher' (recommended for downstream eval) or
    'student'.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(checkpoint_dir)

    ckpt_mngr = ocp.CheckpointManager(checkpoint_dir, ocp.PyTreeCheckpointer())
    step = ckpt_mngr.latest_step()
    if step is None:
        raise FileNotFoundError(f"no checkpoints in {checkpoint_dir}")
    restored = ckpt_mngr.restore(step)

    key = f"{source}_params"
    if key not in restored:
        raise KeyError(
            f"checkpoint missing '{key}'; available: {list(restored.keys())}"
        )

    backbone_params = restored[key]["params"]["backbone"]
    return {"params": backbone_params}, step


# ---------- feature extraction --------------------------------------------

def extract_features(backbone, vit_variables, inputs, n_last_blocks,
                     avgpool_patchtokens):
    """Run the backbone in eval mode and produce a [N, feat_dim] tensor.

    Concatenates the CLS tokens from the last ``n_last_blocks`` blocks. If
    ``avgpool_patchtokens`` is True, the mean-pooled patch tokens of the
    *final* block are concatenated as well (standard DINO linear-eval recipe).
    """
    out = backbone.apply(vit_variables, inputs, masks=None, train=False)

    # [depth, N, E]
    per_block_cls = out["per_block_norm_cls"]
    last_n = per_block_cls[-n_last_blocks:]                 # [n, N, E]
    last_n = jnp.transpose(last_n, (1, 0, 2))               # [N, n, E]
    feats = jnp.reshape(last_n, (last_n.shape[0], -1))      # [N, n*E]

    if avgpool_patchtokens:
        # Mean over patch tokens of the final block (already LayerNormed).
        patch = out["norm_patch_tokens"]                    # [N, L, E]
        patch_mean = jnp.mean(patch, axis=1)                # [N, E]
        feats = jnp.concatenate([feats, patch_mean], axis=-1)

    return feats.astype(jnp.float32)


# ---------- pmapped train / eval steps ------------------------------------

def make_update_fn(*, backbone, classifier_apply_fn, optimizer,
                   n_last_blocks, avgpool_patchtokens):
    def update_fn(params, vit_variables, opt_state, inputs, labels):
        features = extract_features(
            backbone, vit_variables, inputs,
            n_last_blocks=n_last_blocks,
            avgpool_patchtokens=avgpool_patchtokens,
        )
        features = jax.lax.stop_gradient(features)

        def loss_fn(p):
            logits = classifier_apply_fn(p, features)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            return loss, logits

        (loss, logits), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)

        loss = jax.lax.pmean(loss, axis_name="batch")
        grad = jax.lax.pmean(grad, axis_name="batch")

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, logits

    return jax.pmap(update_fn, axis_name="batch", donate_argnums=(0, 2))


def make_predict_fn(*, backbone, classifier_apply_fn, n_last_blocks,
                    avgpool_patchtokens):
    def predict_fn(params, vit_variables, inputs):
        features = extract_features(
            backbone, vit_variables, inputs,
            n_last_blocks=n_last_blocks,
            avgpool_patchtokens=avgpool_patchtokens,
        )
        return classifier_apply_fn(params, features)

    return jax.pmap(predict_fn, axis_name="batch")


def main(config_path):
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)
    print(config)

    dino_config = config["model"]
    dataset_config = config["dataset_params"]
    wandb_config = config["wandb"]

    seed = dino_config["seed"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ])

    if dataset_config["dataset"] == "imagenet":
        train_dataset = ImageFolder(
            root=dataset_config["train_data_path"],
            transform=train_transform,
        )
        val_dataset = ImageFolder(
            root=dataset_config["val_data_path"],
            transform=val_transform,
        )
    else:
        raise ValueError(
            f"unsupported dataset: {dataset_config['dataset']}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        num_workers=dataset_config["num_workers"],
        pin_memory=False,
        drop_last=True,
        prefetch_factor=dataset_config["prefetch_factor"],
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    student_kwargs = dict(dino_config["student_params"])  # copy
    n_last_blocks = student_kwargs.pop("n_last_blocks", 4)
    avgpool_patchtokens = student_kwargs.pop("avgpool_patchtokens", False)

    backbone = vit.__dict__[dino_config["arch"]](**student_kwargs)

    feat_dim = backbone.embed_dim * n_last_blocks
    if avgpool_patchtokens:
        feat_dim += backbone.embed_dim

    classifier = nn.Dense(
        dino_config["num_classes"],
        kernel_init=nn.initializers.truncated_normal(stddev=0.01),
        bias_init=nn.initializers.zeros,
    )

    epochs = dino_config["linear_epochs"]

    run = wandb.init(
        project=wandb_config["project"],
        name="LinearEval",
        reinit=True,
        config=config,
    )

    # ---- load pretrained backbone (Orbax) --------------------------------
    pretrained_dir = dino_config["checkpoint_path"]
    vit_variables, src_step = load_pretrained_backbone_params(
        pretrained_dir, source=dino_config.get("eval_source", "teacher")
    )
    print(f"loaded pretrained backbone from step {src_step}")

    # ---- linear eval head ------------------------------------------------
    linear_eval_checkpoint_path = dino_config["linear_eval_checkpoint_path"]

    key = jax.random.PRNGKey(seed)
    params = classifier.init(key, jnp.ones((2, feat_dim)))

    init_lr = (
        dino_config["optim_params"]["learning_rate"]
        * dataset_config["batch_size"]
        / 256.0
    )
    optimizer = optax.sgd(learning_rate=init_lr, momentum=0.9, nesterov=False)
    opt_state = optimizer.init(params)

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    update_fn = make_update_fn(
        backbone=backbone,
        classifier_apply_fn=classifier.apply,
        optimizer=optimizer,
        n_last_blocks=n_last_blocks,
        avgpool_patchtokens=avgpool_patchtokens,
    )
    predict_fn = make_predict_fn(
        backbone=backbone,
        classifier_apply_fn=classifier.apply,
        n_last_blocks=n_last_blocks,
        avgpool_patchtokens=avgpool_patchtokens,
    )

    params_repl = replicate(params)
    vit_variables_repl = replicate(vit_variables)
    opt_state_repl = replicate(opt_state)

    state_template = {
        "params": unreplicate(params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
    }

    del params, opt_state

    loaded = load_eval_state(linear_eval_checkpoint_path, state_template)
    start_epoch = 0
    if loaded is not None:
        params_repl = replicate(loaded["params"])
        opt_state_repl = replicate(loaded["opt_state"])
        start_epoch = loaded["epoch"] + 1
        print(f"resumed linear-eval head from epoch {loaded['epoch']}")

    def shard(x):
        n, *s = x.shape
        return np.reshape(
            x, (jax.local_device_count(), n // jax.local_device_count(), *s)
        )

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))

    for epoch in range(start_epoch, epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = shard(np.array(images))
            labels = shard(np.array(labels))

            params_repl, opt_state_repl, loss, logits = update_fn(
                params_repl, vit_variables_repl, opt_state_repl,
                images, labels,
            )

            loss_val = float(unreplicate(loss))
            logits_unshard = unshard(logits)
            labels_unshard = unshard(labels)
            preds = jnp.argmax(logits_unshard, axis=-1)
            acc = float(jnp.mean(preds == labels_unshard))

            if step % 50 == 0:
                print(f"epoch {epoch} step {step}  loss {loss_val:.4f}  acc {acc:.4f}")
            run.log({"loss": loss_val, "train_accuracy": acc, "epoch": epoch})

        # ---- validation --------------------------------------------------
        correct = 0
        total = 0
        for images, labels in val_loader:
            images_np = np.array(images)
            labels_np = np.array(labels)
            images_sharded = shard(images_np)
            logits_repl = predict_fn(params_repl, vit_variables_repl,
                                     images_sharded)
            logits = np.asarray(unshard(logits_repl))
            preds = logits.argmax(axis=-1)
            correct += int((preds == labels_np).sum())
            total += int(labels_np.shape[0])

        val_acc = correct / max(total, 1)
        print(f"epoch {epoch} VAL acc {val_acc:.4f}")
        run.log({"val_accuracy": val_acc, "epoch": epoch})

        save_eval_state(linear_eval_checkpoint_path, {
            "params": unreplicate(params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch,
        })


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError("you must provide config file")
    main(sys.argv[1])