import jax
import jax.numpy as jnp

import flax.linen as nn


class WNLinear(nn.Module):
    out_dim: int
    norm_last_layer: bool = True

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]

        v = self.param(
            "v",
            nn.initializers.truncated_normal(stddev=0.02),
            (in_dim, self.out_dim)
        )

        if self.norm_last_layer:
            g = 1.0
        else:
            g = self.param("g", nn.initializers.ones, (self.out_dim,))

        v_norm = v / (jnp.linalg.norm(v, axis=0, keepdims=True) + 1e-12)
        w = v_norm * g

        return jnp.dot(x, w)


class Head(nn.Module):
    out_dim: int = 8192
    norm_last_layer: bool = True
    hidden_dim: int = 2048
    bottleneck_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.bottleneck_dim, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)

        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

        wnl = WNLinear(self.out_dim, self.norm_last_layer)(x)

        return wnl


class DINO(nn.Module):
    backbone: nn.Module
    dino_head: nn.Module
    ibot_head: nn.Module
    num_global_crops: int = 2
    num_local_crops: int = 8
    apply_dino: bool = True
    apply_ibot: bool = True

    @nn.compact
    def __call__(self, x_list, masks=None, train=True):
        global_crops = jnp.concatenate(x_list[:self.num_global_crops], axis=0)
        global_out = self.backbone(global_crops, masks=masks, train=train)

        if global_out.ndim == 3 and self.apply_ibot:  # [N, L, E]
            global_cls = global_out[:, 0]
            global_patches = global_out[:, 1:]
        else:
            global_cls = global_out  # [N, E]
            global_patches = None

        if len(x_list) > self.num_global_crops:
            local_crops = jnp.concatenate(x_list[self.num_global_crops:], axis=0)
            local_out = self.backbone(local_crops, train=train)
            local_cls = local_out[:, 0]

            all_cls = jnp.concatenate([global_cls, local_cls], axis=0)
        else:
            all_cls = global_cls

        out_dict = {}

        if self.apply_dino:
            out_dict['cls'] = self.dino_head(all_cls)

        if self.apply_ibot and global_patches is not None:
            flattened_patches = global_patches.reshape(-1, global_patches.shape[-1])
            out_dict['patch'] = self.ibot_head(flattened_patches)

        return out_dict
