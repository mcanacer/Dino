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
from augmentation import DataAugmentationDINO
import vit
from dino import DINO, DINOHead

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
        center_momentum,
        num_global_crops):
    def update_fn(student_params, teacher_params, opt_state, inputs_list, center, momentum_schedule, rng, step):
        def loss_fn(params):
            student_output = student_apply_fn(
                {'params': params},
                inputs_list,
                train=True,
                rngs={"dropout": rng}
            )

            teacher_output = teacher_apply_fn(
                {'params': teacher_params},
                inputs_list[:num_global_crops],
                train=True
            )

            student_out = student_output / student_temp
            teacher_temp = teacher_temp_fn(step)
            teacher_out = jax.nn.softmax((teacher_output - center) / teacher_temp, axis=-1)

            B = inputs_list[0].shape[0]
            n_student_crops = len(inputs_list)

            student_out_reshaped = student_out.reshape(n_student_crops, B, -1)
            teacher_out_reshaped = teacher_out.reshape(num_global_crops, B, -1)

            teacher_out_reshaped = jax.lax.stop_gradient(teacher_out_reshaped)

            student_log_probs = jax.nn.log_softmax(student_out_reshaped, axis=-1)

            dot_products = jnp.einsum('tbc, sbc -> ts', teacher_out_reshaped, student_log_probs)

            batch_size = B
            loss_matrix = -dot_products / batch_size

            n_t = teacher_out_reshaped.shape[0]
            n_s = student_out_reshaped.shape[0]

            identity = jnp.eye(n_t, n_s)
            mask = 1.0 - identity

            loss = jnp.sum(loss_matrix * mask) / jnp.sum(mask)

            return loss, student_out_reshaped, teacher_out_reshaped

        ((loss, s_out, t_out), grad) = jax.value_and_grad(loss_fn, has_aux=True)(student_params)

        loss = jax.lax.pmean(loss, axis_name='batch')
        grad = jax.lax.pmean(grad, axis_name='batch')

        updates, opt_state = optimizer.update(grad, opt_state, student_params)
        new_student_params = optax.apply_updates(student_params, updates)

        ema_decay = momentum_schedule(step)
        new_teacher_params = ema_update(teacher_params, student_params, decay=ema_decay)

        local_center_mean = jnp.mean(t_out, axis=(0, 1))
        global_center_mean = jax.lax.pmean(local_center_mean, axis_name='batch')
        new_center = center * center_momentum + (1 - center_momentum) * global_center_mean

        return new_student_params, new_teacher_params, opt_state, new_center, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        pin_memory=False,
        drop_last=True,
    )

    student = DINO(
        backbone=vit.__dict__[dino_config['arch']](**dino_config['student_params']),
        head=DINOHead(**dino_config['head_params']),
        num_global_crops=dataset_config['num_global_crops'],
        num_local_crops=dataset_config['num_local_crops'],
    )

    teacher = DINO(
        backbone=vit.__dict__[dino_config['arch']](**dino_config['teacher_params']),
        head=DINOHead(**dino_config['head_params']),
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

    inputs, _ = next(iter(train_loader))

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    student_params = student.init(init_key, jax.tree_util.tree_map(lambda x: np.array(x), inputs), train=True)

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

    teacher_temp_fn = maske_teacher_temp_fn(
        warmup_teacher_temp=dino_config['dino']['warmup_teacher_temp'],
        teacher_temp=dino_config['dino']['teacher_temp'],
        warmup_teacher_temp_epochs=dino_config['dino']['warmup_teacher_temp_epochs'],
        epochs=epochs,
    )

    update_fn = make_update_fn(
        student_apply_fn=student.apply,
        teacher_apply_fn=teacher.apply,
        optimizer=optimizer,
        student_temp=dino_config['dino']['student_temp'],
        teacher_temp_fn=teacher_temp_fn,
        center_momentum=dino_config['dino']['center_momentum'],
        num_global_crops=dataset_config['num_global_crops'],
    )

    student_params_repl = replicate(student_params)
    teacher_params_repl = replicate(teacher_params)
    opt_state_repl = replicate(opt_state)
    center_repl = replicate(center)

    state_template = {
        "student_params": unreplicate(student_params_repl),
        "teacher_params": unreplicate(teacher_params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
    }

    del student_params
    del teacher_params
    del opt_state
    del center

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    global_step = 0

    if loaded_state:
        student_params_repl = replicate(loaded_state['student_params'])
        teacher_params_repl = replicate(loaded_state['teacher_params'])
        opt_state_repl = replicate(loaded_state['opt_state'])
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
        for step, (images, _) in enumerate(train_loader):
            key, dropout_key = jax.random.split(key)
            rng_shard = jax.random.split(dropout_key, jax.local_device_count())

            inputs_list = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)

            (
                student_params_repl,
                teacher_params_repl,
                opt_state_repl,
                center_repl,
                loss
            ) = update_fn(
                student_params_repl,
                teacher_params_repl,
                opt_state_repl,
                inputs_list,
                center_repl,
                momentum_schedule,
                rng_shard,
                step_repl
            )

            step_repl = step_repl + 1

            loss = unreplicate(loss)

            print("Epoch: {} Step: {} Loss: {:.4f}".format(epoch, step, float(loss)))

            run.log({
                "loss": loss,
                "epoch": epoch,
            })

        save_checkpoint(checkpoint_path, {
            "student_params": unreplicate(student_params_repl),
            "teacher_params": unreplicate(teacher_params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch + 1,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
