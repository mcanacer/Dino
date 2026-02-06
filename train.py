import sys
import yaml

import jax
import jax.numpy as jnp
import optax
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
from augmentation import DataAugmentationDINO, Collate
from utils import load_checkpoint, save_checkpoint
import vit
from dino import DINO, Head, Projection
from utils import batch_koleo

import numpy as np


def make_teacher_temp_fn(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, epochs):
    def teacher_temp_fn(step):
        idx = jnp.minimum(step, len(teacher_temp_schedule) - 1)
        return teacher_temp_schedule[idx]
    teacher_temp_schedule = jnp.concatenate((
        jnp.linspace(warmup_teacher_temp,
                    teacher_temp, warmup_teacher_temp_epochs),
        jnp.ones(epochs - warmup_teacher_temp_epochs) * teacher_temp
    ))
    return teacher_temp_fn


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
        momentum_schedule_fn,
        num_crops,
        num_global_crops,
        koleo_loss_weight):
    def update_fn(student_params, teacher_params, opt_state, inputs, masks, rng, step):
        teacher_out_dict = teacher_apply_fn(
            teacher_params,
            inputs,
            is_teacher=True,
            masks=masks,
            train=True,
            tau=teacher_temp_fn(step),
        )

        def loss_fn(params):
            student_out_dict = student_apply_fn(
                params,
                inputs,
                is_teacher=False,
                masks=masks,
                train=True,
                tau=student_temp,
                rngs={"dropout": rng}
            )

            student_cls = student_out_dict['predict_cls']  # [NC*N, E]
            teacher_cls = teacher_out_dict['target_cls']  # [NG*N, E]

            student_patch = student_out_dict.get('predict_patch')  # [NG*N, L, E]
            teacher_patch = teacher_out_dict.get('target_patch')  # [NG*N, L, E]

            g_loss_weight = (num_global_crops - 1) * num_global_crops
            l_loss_weight = max((num_crops - num_global_crops) * num_global_crops , 1)
            dino_total_weight = g_loss_weight + l_loss_weight

            g_loss_weight = g_loss_weight / dino_total_weight
            l_loss_weight = l_loss_weight / dino_total_weight

            student_cls = jnp.reshape(student_cls, (num_crops, -1, student_cls.shape[-1]))  # [NC, N, E]
            teacher_cls = jnp.reshape(teacher_cls, (num_global_crops, -1, teacher_cls.shape[-1]))  # [NG, N, E]

            g_student_cls = student_cls[:num_global_crops]  # [NG, N, E]
            l_student_cls = student_cls[num_global_crops:]  # [NL, N, E]

            l_cls_log_probs = jax.nn.log_softmax(l_student_cls.astype(jnp.float32), axis=-1)  # [NL, N, E]
            l_cls_log_probs = jnp.expand_dims(l_cls_log_probs, axis=1)  # [NL, 1, N, E]

            loss = teacher_cls * l_cls_log_probs  # [NL, NG, N, E]
            loss = jnp.sum(loss, axis=-1)  # [NL, NG, N]
            loss = -jnp.mean(loss)  # [1]

            l_cls_loss = l_loss_weight * loss

            g_cls_log_probs = jax.nn.log_softmax(g_student_cls.astype(jnp.float32), axis=-1)  # [NG, N, E]
            g_cls_log_probs = jnp.expand_dims(g_cls_log_probs, axis=1)  # [NG, 1, N, E]

            loss = teacher_cls * g_cls_log_probs  # [NG, NG, N, E]
            loss = jnp.sum(loss, axis=-1)  # [NG, NG, N]
            loss = jnp.mean(loss, axis=-1)  # [NG, NG]

            mask = 1 - jnp.eye(num_global_crops)  # [NG, NG]
            loss = -jnp.sum(mask * loss)  # [1]

            g_cls_loss = g_loss_weight * loss

            student_patch = jnp.reshape(student_patch,
                                        (num_global_crops, -1, *student_patch.shape[-2:]))  # [NG, N, L, E]
            teacher_patch = jnp.reshape(teacher_patch,
                                        (num_global_crops, -1, *teacher_patch.shape[-2:]))  # [NG, N, L, E]

            patch_log_probs = jax.nn.log_softmax(student_patch.astype(jnp.float32), axis=-1)  # [NG, N, L, E]

            loss = teacher_patch * patch_log_probs  # [NG, N, L, E]
            loss = jnp.sum(loss, axis=-1)  # [NG, N, L]

            masks = jnp.reshape(masks, (masks.shape[0], num_global_crops, -1))  # [N, NG, L]
            masks = jnp.swapaxes(masks, 0, 1)  # [NG, N, L]

            loss = jnp.sum(masks * loss, axis=-1)  # [NG, N]

            mask_counts = jnp.sum(masks, axis=-1)  # [NG, N]
            count_mask = mask_counts != 0  # [NG, N]
            loss /= jnp.where(count_mask, mask_counts, jnp.ones(()))  # [NG, N]
            loss = -jnp.sum(loss)  # [1]
            loss /= jnp.sum(count_mask)  # [1]

            ibot_loss = loss

            g_raw_cls_tokens = student_out_dict["norm_cls_tokens"]  # [NG*N, E]
            g_raw_cls_tokens = jnp.reshape(g_raw_cls_tokens,
                                           (num_global_crops, -1, g_raw_cls_tokens.shape[-1]))  # [NG, N, E]

            loss = batch_koleo(g_raw_cls_tokens.astype(jnp.float32))  # [NG, N]
            loss = -jnp.mean(loss)  # [1]

            koleo_loss = koleo_loss_weight *  loss

            losses = dict(
                g_cls_loss=g_cls_loss,
                l_cls_loss=l_cls_loss,
                ibot_loss=ibot_loss,
                koleo_loss=koleo_loss,
            )

            total_loss = g_cls_loss + l_cls_loss + ibot_loss + koleo_loss
            total_loss = total_loss.astype(jnp.float32)

            return total_loss, losses

        ((loss, losses), grad) = jax.value_and_grad(loss_fn, has_aux=True)(student_params)

        loss = jax.lax.pmean(loss, axis_name='batch')
        grad = jax.lax.pmean(grad, axis_name='batch')

        updates, opt_state = optimizer.update(grad, opt_state, student_params)
        new_student_params = optax.apply_updates(student_params, updates)

        decay = momentum_schedule_fn(step)
        new_teacher_params = jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p,
            teacher_params,
            new_student_params
        )

        return new_student_params, new_teacher_params, opt_state, loss, losses

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0, 1, 2))


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
        dino_proj=Projection(**dino_config['proj_params']),
        ibot_proj=Projection(**dino_config['proj_params']),
        dino_head=Head(**dino_config['head_params']),
        ibot_head=Head(**dino_config['head_params']),
        num_global_crops=dataset_config['num_global_crops'],
        num_local_crops=dataset_config['num_local_crops'],
    )

    teacher = DINO(
        backbone=vit.__dict__[dino_config['arch']](**dino_config['teacher_params']),
        dino_proj=Projection(**dino_config['proj_params']),
        ibot_proj=Projection(**dino_config['proj_params']),
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

    teacher_temp_fn = make_teacher_temp_fn(
        warmup_teacher_temp=dino_config['warmup_teacher_temp'],
        teacher_temp=dino_config['teacher_temp'],
        warmup_teacher_temp_epochs=dino_config['warmup_teacher_temp_epochs'],
        epochs=epochs,
    )

    update_fn = make_update_fn(
        student_apply_fn=student.apply,
        teacher_apply_fn=teacher.apply,
        optimizer=optimizer,
        student_temp=dino_config['student_temp'],
        teacher_temp_fn=teacher_temp_fn,
        momentum_schedule_fn=momentum_schedule,
        num_crops=dataset_config['num_global_crops'] + dataset_config['num_local_crops'],
        num_global_crops=dataset_config['num_global_crops'],
    )

    student_params_repl = replicate(student_params)
    teacher_params_repl = replicate(teacher_params)
    opt_state_repl = replicate(opt_state)

    state_template = {
        "student_params": unreplicate(student_params_repl),
        "teacher_params": unreplicate(teacher_params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
        "key": key,
    }

    del student_params
    del teacher_params
    del opt_state

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    global_step = 0

    if loaded_state:
        student_params_repl = replicate(loaded_state['student_params'])
        teacher_params_repl = replicate(loaded_state['teacher_params'])
        opt_state_repl = replicate(loaded_state['opt_state'])
        start_epoch = loaded_state['epoch'] + 1
        key = loaded_state['key']
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
                loss,
                losses
            ) = update_fn(
                student_params_repl,
                teacher_params_repl,
                opt_state_repl,
                images,
                masks,
                rng_shard,
                step_repl
            )

            step_repl = step_repl + 1

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "g_cls_loss": losses['g_cls_loss'],
                "l_cls_loss": losses['l_cls_loss'],
                "ibot_loss": losses['ibot_loss'],
                "koleo_loss": losses['koleo_loss'],
                "epoch": epoch,
            })

        save_checkpoint(checkpoint_path, {
            "student_params": unreplicate(student_params_repl),
            "teacher_params": unreplicate(teacher_params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch + 1,
            "key": key,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
