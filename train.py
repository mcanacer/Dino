import sys
import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
from augmentation import DataAugmentationDINO, Collate
import vit
from dino import DINO, Head

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def maske_teacher_temp_fn(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, epochs):
    def teacher_temp_fn(step):
        idx = jnp.minimum(step, len(teacher_temp_schedule) - 1)
        return teacher_temp_schedule[idx]
    teacher_temp_schedule = jnp.concatenate((
        jnp.linspace(warmup_teacher_temp,
                    teacher_temp, warmup_teacher_temp_epochs),
        jnp.ones(epochs - warmup_teacher_temp_epochs) * teacher_temp
    ))
    return teacher_temp_fn


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def create_weight_decay_mask(params):
    def _is_kernel(path, leaf):
        key_name = path[-1].key
        return key_name == 'kernel'
    return jax.tree_util.tree_map_with_path(_is_kernel, params)


def create_dino_scheduler(base_value, final_value, epochs, niter_per_epoch, warmup_epochs=0, start_warmup_value=0):
    def schedule_fn(step):
        idx = jnp.minimum(step, len(schedule) - 1)
        return schedule[idx]
    warmup_iters = warmup_epochs * niter_per_epoch
    total_iters = epochs * niter_per_epoch

    warmup_schedule = jnp.linspace(start_warmup_value, base_value, warmup_iters)

    iters = jnp.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = jnp.concatenate([warmup_schedule, schedule])
    return schedule_fn


def make_update_fn(
        student_apply_fn,
        teacher_apply_fn,
        optimizer,
        student_temp,
        teacher_temp_fn,
        teacher_temp2_fn,
        center_momentum,
        momentum_schedule_fn,
        num_crops,
        num_global_crops):
    def update_fn(student_params, teacher_params, opt_state, inputs_list, masks, center, center2, rng, step):
        def loss_fn(params):
            student_out_dict = student_apply_fn(
                params,
                inputs_list,
                masks=masks,
                train=True,
                rngs={"dropout": rng}
            )

            teacher_out_dict = teacher_apply_fn(
                teacher_params,
                inputs_list[:num_global_crops],
                train=True
            )

            student_cls = student_out_dict['cls']  # [Num_Crops*N, E]
            teacher_cls = teacher_out_dict['cls']  # [Num_Global_Crops*N, E]

            student_patch = student_out_dict.get('patch')  # [Num_Global_Crops*N, L, E]
            teacher_patch = teacher_out_dict.get('patch')  # [Num_Global_Crops*N, L, E]

            student_cls = student_cls / student_temp  # [Num_Crops*N, E]
            student_cls_logits = student_cls.chunk(num_crops)  # [N, E] for each

            teacher_temp = teacher_temp_fn(step)
            teacher_cls_centered = teacher_cls - center  # [Num_Global_Crops*N, E]
            teacher_cls_probs = jax.nn.softmax(teacher_cls_centered / teacher_temp, axis=-1)  # [Num_Global_Crops*N, E]
            teacher_cls_probs_chunked = teacher_cls_probs.chunk(num_global_crops)  # [N, E] for each

            N = inputs_list[0].shape[0]

            student_cls_logits_stacked = jnp.stack(student_cls_logits)  # [Num_Local_Crops, N, E]
            teacher_cls_probs_stacked = jnp.stack(teacher_cls_probs_chunked)  # [Num_Global_Crops, N, E]

            student_log_probs = jax.nn.log_softmax(student_cls_logits_stacked, axis=-1)  # [Num_Local_Crops, N, E]

            # [Num_Global_Crops, Num_Crops]
            dot_products = jnp.einsum('tne, sne -> ts', teacher_cls_probs_stacked, student_log_probs)

            dino_loss_matrix = -dot_products / N  # [Num_Global_Crops, Num_Crops]

            identity = jnp.eye(num_global_crops, num_crops)  # [2, Num_Crops]
            mask = 1.0 - identity  # [2, Num_Crops]
            dino_loss = jnp.sum(dino_loss_matrix * mask) / jnp.sum(mask)

            ibot_loss = 0.0

            teacher_patch_mean_batch = jnp.zeros_like(center2)  # [1, 1, E]

            if student_patch is not None and teacher_patch is not None:
                student_patch = student_patch / student_temp  # [Num_Global_Crops*N, L, E]
                student_patch = student_patch.reshape(num_global_crops, N, -1, student_patch.shape[-1])  # [2, N, L, E]
                student_log_patch = jax.nn.log_softmax(student_patch, axis=-1)  # [2, N, L, E]

                teacher_temp2 = teacher_temp2_fn(step)
                teacher_patch_centered = teacher_patch - center2  # [Num_Global_Crops*N, L, E]
                teacher_patch_probs = jax.nn.softmax(teacher_patch_centered / teacher_temp2, axis=-1)  # [Num_Global_Crops*N, L, E]
                teacher_patch_probs = teacher_patch_probs.reshape(num_global_crops, N, -1, teacher_patch.shape[-1])  # [2, N, L, E]

                teacher_patch_flat = teacher_patch_probs.reshape(-1, teacher_patch_probs.shape[-1])  # [Num_Global_Crops*N*L, E]
                teacher_patch_mean_batch = jnp.mean(teacher_patch_flat, axis=0, keepdims=True)  # [1, E]

                ce_loss = -jnp.sum(teacher_patch_probs * student_log_patch, axis=-1)  # [2, N, L]

                masks_flat = jnp.transpose(masks, (1, 0, 2, 3))  # [N, 2, H, W]
                masks_flat = masks_flat.reshape(num_global_crops, N, -1)  # [2, N, L]

                masked_ce_loss = masks_flat * ce_loss  # [2, N, L]

                n_masked = jnp.sum(masks_flat) + 1e-6
                ibot_loss = jnp.sum(masked_ce_loss) / n_masked

            total_loss = dino_loss + ibot_loss

            aux_stats = {
                "dino_loss": dino_loss,
                "ibot_loss": ibot_loss,
                "teacher_cls_mean": jnp.mean(teacher_cls_probs, axis=0, keepdims=True),
                "teacher_patch_mean": teacher_patch_mean_batch
            }

            return total_loss, aux_stats

        (loss, aux_stats, grad) = jax.value_and_grad(loss_fn, has_aux=True)(student_params)

        loss = jax.lax.pmean(loss, axis_name='batch')
        grad = jax.lax.pmean(grad, axis_name='batch')

        teacher_cls_mean = jax.lax.pmean(aux_stats['teacher_cls_mean'], axis_name='batch')
        teacher_patch_mean = jax.lax.pmean(aux_stats['teacher_patch_mean'], axis_name='batch')

        updates, opt_state = optimizer.update(grad, opt_state, student_params)
        new_student_params = optax.apply_updates(student_params, updates)

        ema_decay = momentum_schedule_fn(step)
        new_teacher_params = ema_update(teacher_params, student_params, decay=ema_decay)

        new_center = center * center_momentum + (1 - center_momentum) * teacher_cls_mean
        new_center2 = center2 * center_momentum + (1 - center_momentum) * teacher_patch_mean

        return new_student_params, new_teacher_params, opt_state, new_center, new_center2, loss, aux_stats

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0, 1, 2, 5, 6))


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dino_config = config['dino']
    dataset_config = config['dataset_params']
    wandb_config = config['wandb']

    seed = dino_config['seed']

    transform = DataAugmentationDINO(
        dataset_config['global_crops_scale'],
        dataset_config['local_crops_scale'],
        dataset_config['num_local_crops'],
    )

    if dataset_config['dataset'] == 'imagenet':
        train_dataset = ImageFolder(
            root=dataset_config['data_path'],
            transform=transform,
        )
    else:
        raise ValueError('There is no such dataset')

    collate_fn = Collate(
        image_size=dataset_config['image_size'],
        num_global_crops=dataset_config['num_global_crops'],
        patch_size=dataset_config['patch_size'],
        pred_ratio=dataset_config['pred_ratio'],
        min_aspect_ratio=dataset_config['min_aspect_ratio'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=dataset_config['prefetch_factor'],
        persistent_workers=True,
    )

    student = DINO(
        backbone=vit.__dict__[dino_config['arch']](**dino_config['student_params']),
        dino_head=Head(**dino_config['head_params']),
        ibot_head=Head(**dino_config['head_params']),
        num_global_crops=dataset_config['num_global_crops'],
        num_local_crops=dataset_config['num_local_crops'],
    )

    teacher = DINO(
        backbone=vit.__dict__[dino_config['arch']](**dino_config['teacher_params']),
        dino_head=Head(**dino_config['head_params']),
        ibot_head=Head(**dino_config['head_params']),
        num_global_crops=dataset_config['num_global_crops'],
        num_local_crops=dataset_config['num_local_crops'],
    )

    epochs = dino_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = dino_config['checkpoint_path']

    inputs, masks = next(iter(train_loader))
    inputs = [np.array(x) for x in inputs]
    masks = np.array(masks)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    student_params = student.init(init_key, inputs, masks=masks, train=True)

    del inputs, masks

    steps_per_epoch = len(train_loader)

    init_lr = dino_config['optim_params']['lr'] * dataset_config['batch_size'] / 256

    lr_schedule = create_dino_scheduler(
        init_lr,
        dino_config['optim_params']["min_lr"],
        epochs,
        steps_per_epoch,
        warmup_epochs=dino_config['optim_params']['warmup_epochs'],
    )

    wd_schedule = create_dino_scheduler(
        dino_config['optim_params']['weight_decay'],
        dino_config['optim_params']['weight_decay_end'],
        epochs,
        steps_per_epoch,
    )

    momentum_schedule = create_dino_scheduler(
        dino_config['optim_params']['momentum_teacher'],
        final_value=1,
        epochs=epochs,
        niter_per_epoch=steps_per_epoch,
    )

    wd_mask = create_weight_decay_mask(student_params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(dino_config['optim_params']['grad_clip']),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=wd_schedule,
            mask=wd_mask
        )
    )

    opt_state = optimizer.init(student_params)

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    teacher_params = student_params
    center = jnp.zeros((1, dino_config['head_params']['out_dim']))
    center2 = jnp.zeros((1, 1, dino_config['head_params']['out_dim']))

    teacher_temp_fn = maske_teacher_temp_fn(
        warmup_teacher_temp=dino_config['warmup_teacher_temp'],
        teacher_temp=dino_config['teacher_temp'],
        warmup_teacher_temp_epochs=dino_config['warmup_teacher_temp_epochs'],
        epochs=epochs,
    )

    teacher_temp2_fn = maske_teacher_temp_fn(
        warmup_teacher_temp=dino_config['warmup_teacher_temp2'],
        teacher_temp=dino_config['teacher_temp2'],
        warmup_teacher_temp_epochs=dino_config['warmup_teacher_patch_temp'],
        epochs=epochs,
    )

    update_fn = make_update_fn(
        student_apply_fn=student.apply,
        teacher_apply_fn=teacher.apply,
        optimizer=optimizer,
        student_temp=dino_config['student_temp'],
        teacher_temp_fn=teacher_temp_fn,
        teacher_temp2_fn=teacher_temp2_fn,
        center_momentum=dino_config['center_momentum'],
        momentum_schedule_fn=momentum_schedule,
        num_crops=dataset_config['num_global_crops'] + dataset_config['num_local_crops'],
        num_global_crops=dataset_config['num_global_crops'],
    )

    student_params_repl = replicate(student_params)
    teacher_params_repl = replicate(teacher_params)
    opt_state_repl = replicate(opt_state)
    center_repl = replicate(center)
    center2_repl = replicate(center2)

    state_template = {
        "student_params": unreplicate(student_params_repl),
        "teacher_params": unreplicate(teacher_params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "center": unreplicate(center_repl),
        "center2": unreplicate(center2_repl),
        "epoch": 0,
    }

    del student_params
    del teacher_params
    del opt_state
    del center
    del center2

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    global_step = 0

    if loaded_state:
        student_params_repl = replicate(loaded_state['student_params'])
        teacher_params_repl = replicate(loaded_state['teacher_params'])
        opt_state_repl = replicate(loaded_state['opt_state'])
        center_repl = loaded_state['center']
        center2_repl = loaded_state['center2']
        start_epoch = loaded_state['epoch'] + 1
        global_step = steps_per_epoch * start_epoch

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (jax.local_device_count(), n // jax.local_device_count(), *s))

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))

    step_repl = replicate(global_step)

    for epoch in range(start_epoch, epochs):
        for step, (images, masks) in enumerate(train_loader):
            key, dropout_key = jax.random.split(key)
            rng_shard = jax.random.split(dropout_key, jax.local_device_count())

            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            masks = jax.tree_util.tree_map(lambda x: shard(np.array(x)), masks) if masks is not None else None

            (
                student_params_repl,
                teacher_params_repl,
                opt_state_repl,
                center_repl,
                center2_repl,
                loss,
                aux_stats
            ) = update_fn(
                student_params_repl,
                teacher_params_repl,
                opt_state_repl,
                images,
                masks,
                center_repl,
                center2_repl,
                rng_shard,
                step_repl
            )

            step_repl = step_repl + 1

            loss = unreplicate(loss)
            dino_loss = unreplicate(aux_stats['dino_loss'])
            ibot_loss = unreplicate(aux_stats['ibot_loss'])

            print("Epoch: {} Step: {} Loss: {:.4f} Dino Loss: {:.4f} iBOT Loss{:.4f}".format(
                epoch, step, float(loss), float(dino_loss), float(ibot_loss)))

            run.log({
                "loss": loss,
                "dino_loss": dino_loss,
                "ibot_loss": ibot_loss,
                "epoch": epoch,
            })

        save_checkpoint(checkpoint_path, {
            "student_params": unreplicate(student_params_repl),
            "teacher_params": unreplicate(teacher_params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "center": unreplicate(center_repl),
            "center2": unreplicate(center2_repl),
            "epoch": epoch + 1,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
