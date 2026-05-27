import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import orbax.checkpoint as ocp

from torchvision import transforms

import vit


def _get_norm_input_to_block(model, params, inputs, block_idx):
    _, variables = model.apply(
        params,
        inputs,
        masks=None,
        train=False,
        capture_intermediates=True,
        mutable=["intermediates"],
    )

    block_key = f"Block_{block_idx}"
    block_intermediates = variables["intermediates"][block_key]

    ln_out = block_intermediates["LayerNorm_0"]["__call__"][0]
    return ln_out


def _qk_params_for_block(params, block_idx):
    block_params = params["params"][f"Block_{block_idx}"]["MultiHeadAttention_0"]
    return block_params["query"], block_params["key"]


def visualize_specific_image(
    model,
    params,
    img_path,
    block_idx=11,
    img_size=224,
    save_path=None,
):
    raw_img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),  # FIX: force square
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # CHW -> HWC
        ]
    )
    img_tensor = transform(raw_img).unsqueeze(0)
    inputs = jnp.array(img_tensor.numpy())

    ln_out = _get_norm_input_to_block(model, params, inputs, block_idx)

    q_params, k_params = _qk_params_for_block(params, block_idx)

    def project(x, dense_params):
        kernel = dense_params["kernel"].astype(jnp.float32)
        out = jnp.einsum("btf,fhd->bthd", x.astype(jnp.float32), kernel)
        if "bias" in dense_params:
            bias = dense_params["bias"].astype(jnp.float32)  # [num_heads, head_dim]
            out = out + bias
        return out

    q = project(ln_out, q_params)  # [1, T, H, D]
    k = project(ln_out, k_params)  # [1, T, H, D]

    head_dim = q.shape[-1]
    attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
    attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1)

    num_total_tokens = q.shape[1]
    num_registers = getattr(model, "num_registers", 0)
    num_patches = num_total_tokens - 1 - num_registers
    grid_size = int(round(np.sqrt(num_patches)))
    assert grid_size * grid_size == num_patches, (
        f"non-square patch count {num_patches}; check img_size/patch_size"
    )

    patch_start = 1 + num_registers
    cls_attn = attn_weights[0, :, 0, patch_start:]  # [num_heads, num_patches]
    num_heads = cls_attn.shape[0]
    cls_attn_grid = np.array(cls_attn).reshape(num_heads, grid_size, grid_size)

    img_array = np.array(raw_img.resize((img_size, img_size))) / 255.0
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(3 * (num_heads + 1), 4))
    axes[0].imshow(img_array)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    for i in range(num_heads):
        ax = axes[i + 1]
        m = cls_attn_grid[i]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        ax.imshow(m, cmap="magma")
        ax.set_title(f"Head {i}")
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved attention map to {save_path}")
    else:
        plt.show()


def load_backbone_params_from_orbax(checkpoint_dir, step=None):
    ckpt_mngr = ocp.CheckpointManager(
        checkpoint_dir,
        ocp.PyTreeCheckpointer(),
    )
    if step is None:
        step = ckpt_mngr.latest_step()
        if step is None:
            raise FileNotFoundError(f"no checkpoints in {checkpoint_dir}")

    restored = ckpt_mngr.restore(step)
    teacher_params_full = restored["teacher_params"]
    backbone_params = teacher_params_full["params"]["backbone"]
    return {"params": backbone_params}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="path to input image")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Orbax checkpoint directory (training output)")
    p.add_argument("--arch", type=str, default="vit_small",
                   choices=["vit_tiny", "vit_small", "vit_base"])
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--num_registers", type=int, default=4)
    p.add_argument("--block", type=int, default=-1,
                   help="which block to visualise (-1 = last)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()

    backbone = vit.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=0.0,
        mask_im_modeling=True,
        num_registers=args.num_registers,
    )

    if args.checkpoint_dir is not None and os.path.isdir(args.checkpoint_dir):
        params = load_backbone_params_from_orbax(args.checkpoint_dir)
    else:
        print("WARNING: no checkpoint provided, using random params.")
        params = backbone.init(
            jax.random.PRNGKey(args.seed),
            jnp.ones((1, args.img_size, args.img_size, 3)),
            train=False,
        )

    block_idx = args.block if args.block >= 0 else backbone.depth - 1
    visualize_specific_image(
        backbone, params, args.image,
        block_idx=block_idx,
        img_size=args.img_size,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()