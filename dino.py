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


class DINOHead(nn.Module):
    out_dim: int = 65536
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
    head: nn.Module
    num_global_crops: int = 2
    num_local_crops: int = 8

    @nn.compact
    def __call__(self, x_list, train=True):
        global_crops = jnp.concatenate(x_list[:self.num_global_crops], axis=0)
        global_out = self.backbone(global_crops, train=train)

        if len(x_list) > self.num_global_crops:
            local_crops = jnp.concatenate(x_list[self.num_global_crops:], axis=0)
            local_out = self.backbone(local_crops, train=train)

            output_feats = jnp.concatenate([global_out, local_out], axis=0)
        else:
            output_feats = global_out

        return self.head(output_feats)
