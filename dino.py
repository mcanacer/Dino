import jax
import jax.numpy as jnp

import flax.linen as nn
from utils import sinkhorn_knopp, l2


class Projection(nn.Module):
    hidden_dim: int
    bottleneck_dim: int
    n_layers: int = 3
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = jax.nn.gelu(x)
        for _ in range(self.n_layers - 2):
            x = nn.Dense(
                features=self.hidden_dim,
                dtype=self.dtype,
                param_dtype=jnp.float32,
            )(x)
            x = jax.nn.gelu(x)

        x = nn.Dense(
            features=self.bottleneck_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        return x


class Head(nn.Module):
    hidden_dim: int

    use_bias: bool = False
    use_weight_norm: bool = True

    norm_eps: float = 1e-12
    weight_norm_eps: float = 1e-12

    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train=False):
        x = l2(x.astype(jnp.float32), axis=-1, eps=self.norm_eps)

        mdl = nn.Dense(
            features=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.truncated_normal(0.02),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )
        if self.use_weight_norm:
            mdl = nn.WeightNorm(mdl, epsilon=self.weight_norm_eps)
        return mdl(x)


class DINO(nn.Module):
    backbone: nn.Module
    dino_proj: nn.Module
    ibot_proj: nn.Module
    dino_head: nn.Module
    ibot_head: nn.Module
    num_global_crops: int = 2
    num_local_crops: int = 8
    apply_dino: bool = True
    apply_ibot: bool = True

    @nn.compact
    def __call__(self, x_list, is_teacher=False, masks=None, train=True, tau=1.0):
        if is_teacher and self.apply_dino:
            global_crops = jnp.concatenate(x_list[:self.num_global_crops], axis=0)
            global_out = self.backbone(global_crops, train=False)

            teacher_cls = global_out['norm_cls_tokens']  # [NG*N, E]

            teacher_cls = self.dino_proj(teacher_cls)
            teacher_cls = self.dino_head(teacher_cls)

            global_out["target_cls"] = sinkhorn_knopp(
                (teacher_cls / tau).astype(jnp.float32),
                axis_name="batch" if train else None,
            )

            if self.apply_ibot:
                teacher_patch = global_out["norm_patch_tokens"]

                teacher_patch = self.ibot_proj(teacher_patch)
                teacher_patch = self.ibot_head(teacher_patch)

                global_out["target_patch"] = sinkhorn_knopp(
                    (teacher_patch / tau).astype(jnp.float32),
                    mask=masks.transpose(1, 0, 2),  # [N, NG, L] -> [NG, N, L]
                    axis_name="batch" if train else None,
                )

            return global_out
        else:
            global_crops = jnp.concatenate(x_list[:self.num_global_crops], axis=0)
            global_out = self.backbone(global_crops, masks=masks, train=train)

            student_cls = global_out['norm_cls_tokens']
            student_patch = global_out['norm_patch_tokens'] if self.apply_ibot else None
            if len(x_list) > self.num_global_crops:
                local_crops = jnp.concatenate(x_list[self.num_global_crops:], axis=0)
                local_out = self.backbone(local_crops, train=train)
                local_cls = local_out['norm_cls_tokens']

                student_cls = jnp.concatenate([student_cls, local_cls], axis=0)

            student_cls = self.dino_proj(student_cls)
            student_cls = self.dino_head(student_cls)

            if self.apply_ibot:
                assert student_patch is not None
                student_patch = self.ibot_proj(student_patch)
                student_patch = self.ibot_head(student_patch)

            student_cls = student_cls / tau
            student_patch = student_patch / tau if student_patch is not None else None

            global_out["predict_cls"] = student_cls
            global_out["predict_patch"] = student_patch

            return global_out
