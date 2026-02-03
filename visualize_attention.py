import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import vit
from torchvision import datasets, transforms


def visualize_specific_image(model, params, img_path):
    raw_img = Image.open(img_path).convert('RGB')
    img_array = np.array(raw_img) / 255.0

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ])
    img = transform(raw_img)
    inputs = img.unsqueeze(0)
    inputs = jnp.array(inputs)

    _, variables = model.apply(
        params,
        inputs,
        masks=None,
        train=False,
        capture_intermediates=True,
        mutable=["intermediates"],
    )

    q = variables["intermediates"]["Block_11"]["MultiHeadAttention_0"]["query"]["__call__"][0]
    k = variables["intermediates"]["Block_11"]["MultiHeadAttention_0"]["key"]["__call__"][0]

    head_dim = q.shape[-1]
    attn_logits = jnp.einsum("bqhd, bkhd -> bhqk", q, k) / jnp.sqrt(head_dim)
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    num_total_tokens = q.shape[1]
    num_patches = 196
    start_idx = num_total_tokens - num_patches

    cls_attn = attn_weights[0, :, 0, start_idx:]
    num_heads = cls_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))

    cls_attn_grid = cls_attn.reshape(num_heads, grid_size, grid_size)

    fig, axes = plt.subplots(1, num_heads + 1, figsize=(3 * (num_heads + 1), 4))

    axes[0].imshow(img_array)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    for i in range(num_heads):
        ax = axes[i + 1]

        mask = cls_attn_grid[i]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        ax.imshow(mask, cmap="magma")
        ax.set_title(f"Head {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

backbone = vit.__dict__["vit_small"](drop_path_rate=0.1, mask_im_modeling=True)

params = backbone.init(jax.random.PRNGKey(42), jnp.ones((1, 224, 224, 3)), train=False)

visualize_specific_image(backbone, params, "/Users/muhammetcan/PycharmProjects/dino/images/object1/generated_image0.jpeg")
