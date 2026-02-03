import functools

import jax
import jax.numpy as jnp

import os
from flax import serialization


def pairwise_distance(a, b, p=2, eps=1e-8):
    return jnp.pow(jnp.sum(jnp.pow(a - b, p), axis=-1), 1 / (p + eps))


def l2(x, axis=-1, eps=1e-7):
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def sinkhorn_knopp(inputs,
                   mask=None,
                   num_iterations=3,
                   axis_name=None):

    def sum(*args, **kwargs):
        inputs = jnp.sum(*args, **kwargs)
        if axis_name is not None:
            inputs = jax.lax.psum(
                inputs,
                axis_name=axis_name,
            )
        return inputs

    if mask is not None:
        mask = jnp.reshape(mask, (-1, *jnp.shape(mask)[2:]))
        mask = jnp.expand_dims(mask, axis=-1)
        inverse_mask = 1 - mask

    *ns, c = jnp.shape(inputs)
    axis = range(len(ns))

    n = functools.reduce(lambda x, y: x * y, ns, 1)
    if mask is not None:
        n = sum(mask)

    q = jnp.exp(inputs)
    q = q * mask if mask is not None else q
    q /= sum(q)

    for _ in range(num_iterations):
        q = q * mask if mask is not None else q
        q /= sum(q, axis=axis, keepdims=True)
        q /= c

        s = jnp.sum(q, axis=-1, keepdims=True)
        q /= s + inverse_mask if mask is not None else s
        q /= n

    q *= n
    return q


def batch_koleo(inputs, p=2, eps=1e-8):
    n, m, c = jnp.shape(inputs)

    inputs = l2(inputs, axis=-1, eps=eps)

    distances = jnp.einsum('...ab, ...cb -> ...ac', inputs, inputs)
    eye = jnp.eye(m)
    distances *= -eye + (1 - eye)
    indices = jnp.argmax(distances, axis=-1)

    batch_indices = jnp.arange(n) * m
    batch_indices = jnp.expand_dims(batch_indices, axis=-1)
    indices = jnp.reshape(batch_indices + indices, (-1,))

    others = jnp.reshape(inputs, (n * m, c))
    others = jnp.reshape(others[indices], (n, m, c))

    distances = pairwise_distance(inputs, others, p, eps=eps)
    distances = jnp.log(distances + eps)

    return distances

def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())

