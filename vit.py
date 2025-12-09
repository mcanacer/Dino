# Code converted to jax/flax
# https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import math
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


class DropPath(nn.Module):
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=False):
        if deterministic or self.drop_prob == 0.:
            return x
        rng = self.make_rng('dropout')
        return drop_path(x, self.drop_prob, training=not deterministic, rng=rng)


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


class Attention(nn.Module):
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, deterministic=False):
        N, L, C = x.shape
        head_dim = C // self.num_heads
        scale = self.qk_scale or head_dim ** -0.5

        qkv = nn.Dense(C * 3, use_bias=self.qkv_bias, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        qkv = qkv.reshape(N, L, 3, self.num_heads, C // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

        x = (attn @ v).transpose((0, 2, 1, 3)).reshape(N, L, C)
        x = nn.Dense(C, kernel_init=nn.initializers.truncated_normal(stddev=0.02))(x)
        x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)
        return x, attn


class Block(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    act_layer: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic=False, return_attention=False):
        y, attn = Attention(
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.drop
        )(nn.LayerNorm()(x), deterministic=deterministic)

        if return_attention:
            return attn

        if self.drop_path > 0.:
            y = DropPath(drop_prob=self.drop_path)(y, deterministic=deterministic)
        x = x + y

        mlp_hidden_dim = int(x.shape[-1] * self.mlp_ratio)
        mlp_out = Mlp(
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop
        )(nn.LayerNorm()(x), deterministic=deterministic)

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
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.
    mask_im_modeling: bool = False
    return_all_tokens: bool = False

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

    def prepare_tokens(self, x, cls_token, pos_embed, mask_embed, mask=None, deterministic=False):
        N, H, W, C = x.shape
        x = self.patch_embed(x)

        if mask is not None:
            x[mask, :] = mask_embed

        x = x.reshape(N, -1, self.embed_dim)

        cls_tokens = jnp.broadcast_to(cls_token, (N, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        x = x + self.interpolate_pos_encoding(x, W, H, pos_embed)

        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)
        return x

    @nn.compact
    def __call__(self, x, masks=None, train=True):
        if self.mask_im_modeling:
            mask_embed = self.param('mask_embed', nn.initializers.zeros, (1, self.embed_dim))
        else:
            mask_embed = None
        cls_token = self.param('cls_token', nn.initializers.truncated_normal(stddev=0.02), (1, 1, self.embed_dim))
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.num_patches + 1, self.embed_dim)
        )

        x = self.prepare_tokens(x, cls_token, pos_embed, mask_embed, masks, not train)

        dpr = [x for x in np.linspace(0, self.drop_path_rate, self.depth)]

        for i in range(self.depth):
            x = Block(
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i]
            )(x, deterministic=not train)

        x = nn.LayerNorm()(x)

        if self.return_all_tokens:
            return x
        return x[:, 0]


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
