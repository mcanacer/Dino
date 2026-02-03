# Code converted to jax/flax
# https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import math
import typing
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

import numpy as np


def drop_path(x, drop_prob, training=False, rng=None):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + jax.random.uniform(rng, shape, dtype=x.dtype)
    random_tensor = jnp.floor(random_tensor)
    output = x / keep_prob * random_tensor
    return output


class LayerScale(nn.Module):
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        embed_dim = jnp.shape(inputs)[-1]

        scale = self.param('scale', nn.initializers.constant(self.init_values), (embed_dim,))
        return inputs * scale


class DropPath(nn.Module):
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=False):
        if deterministic or self.drop_prob == 0.:
            return x
        rng = self.make_rng('dropout')
        return drop_path(x, self.drop_prob, training=not deterministic, rng=rng)


class SwiGLU(nn.Module):
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable = nn.silu
    drop: float = 0.

    @nn.compact
    def __call__(self, x, deterministic=False):
        in_features = x.shape[-1]
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features

        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        x = nn.Dense(2 * hidden_features, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        x1, x2 = jnp.split(x, 2, axis=-1)
        x = self.act_layer(x1) * x2
        x = nn.Dense(out_features, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        return x


class Mlp(nn.Module):
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable = nn.gelu
    drop: float = 0.

    @nn.compact
    def __call__(self, x, deterministic=False):
        in_features = x.shape[-1]
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features

        x = nn.Dense(hidden_features, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        x = nn.Dense(out_features, kernel_init=nn.initializers.truncated_normal(stddev=0.02),)(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        return x


class Block(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    ffn: typing.Type = SwiGLU
    init_values: float = 1e-5
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    act_layer: Callable = nn.silu
    eps: float = 1e-06

    @nn.compact
    def __call__(self, x, deterministic=False, return_attention=False):
        y = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            use_bias=self.qkv_bias,
            dropout_rate=self.attn_drop,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(nn.LayerNorm(epsilon=self.eps)(x), deterministic=deterministic)

        if self.init_values is not None:
            y = LayerScale(self.init_values)(y)

        if self.drop_path > 0.:
            y = DropPath(drop_prob=self.drop_path)(y, deterministic=deterministic)
        x = x + y

        mlp_hidden_dim = int(x.shape[-1] * self.mlp_ratio)
        mlp_out = self.ffn(
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop
        )(nn.LayerNorm(epsilon=self.eps)(x), deterministic=deterministic)
        if self.init_values is not None:
            mlp_out = LayerScale(self.init_values)(mlp_out)

        if self.drop_path > 0.:
            mlp_out = DropPath(drop_prob=self.drop_path)(mlp_out, deterministic=deterministic)
        x = x + mlp_out
        return x


class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768

    def setup(self):
        self.num_patches = (self.img_size // self.patch_size) ** 2

    @nn.compact
    def __call__(self, x):
        N, H, W, C = x.shape
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID'
        )(x)
        x = x.reshape(N, -1, self.embed_dim)
        return x


class VisionTransformer(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    ffn: typing.Type = SwiGLU
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.
    mask_im_modeling: bool = False
    num_registers: int = 4
    eps: float = 1e-06

    def setup(self):
        self.num_features = self.embed_dim
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.num_patches = num_patches

    def interpolate_pos_encoding(self, x, w, h, pos_embed):
        npatch = x.shape[1] - 1
        L = pos_embed.shape[1] - 1
        if npatch == L and w == h:
            return pos_embed

        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(L)), int(math.sqrt(L)), dim
        )
        patch_pos_embed = jax.image.resize(
            patch_pos_embed,
            (1, int(w0), int(h0), dim),
            method='bicubic'
        )
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        return jnp.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def prepare_tokens(self, x, cls_token, pos_embed, mask_embed, mask=None, register_embed=None):
        N, H, W, C = x.shape
        x = self.patch_embed(x)

        if mask is not None:
            mask = mask.reshape(N, -1, 1)
            x = mask_embed * mask + x * (1 - mask)

        x = x.reshape(N, -1, self.embed_dim)

        cls_tokens = jnp.broadcast_to(cls_token, (N, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        x = x + self.interpolate_pos_encoding(x, W, H, pos_embed)

        if self.num_registers > 0:
            register_embed = jnp.repeat(register_embed, N, axis=0)
            x = jnp.concatenate([x[:, :1], register_embed, x[:, 1:]], axis=1)

        return x

    @nn.compact
    def __call__(self, x, masks=None, train=True):
        if self.mask_im_modeling:
            mask_embed = self.param('mask_embed', nn.initializers.zeros, (1, self.embed_dim))
        else:
            mask_embed = None
        if self.num_registers > 0:
            register_embed = self.param('register_embed',
                                        nn.initializers.truncated_normal(stddev=1e-6),
                                        (1, self.num_registers, self.embed_dim))
        else:
            register_embed = None
        cls_token = self.param('cls_token', nn.initializers.truncated_normal(stddev=1e-6), (1, 1, self.embed_dim))
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.num_patches + 1, self.embed_dim)
        )

        x = self.prepare_tokens(x, cls_token, pos_embed, mask_embed, masks, register_embed)

        dpr = [x for x in np.linspace(0, self.drop_path_rate, self.depth)]

        for i in range(self.depth):
            x = Block(
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                ffn=self.ffn,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i],
                eps=self.eps,
            )(x, deterministic=not train)

        x_norm = nn.LayerNorm(epsilon=self.eps)(x)

        return {
            "cls_tokens": x[:, 0],
            "registers": x[:, 1:self.num_registers + 1],
            "patch_tokens": x[:, self.num_registers + 1:],
            "norm_cls_tokens": x_norm[:, 0],
            "norm_registers": x_norm[:, 1:self.num_registers + 1],
            "norm_patch_tokens": x_norm[:, self.num_registers + 1:],
        }


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model
