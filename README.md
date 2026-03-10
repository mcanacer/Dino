# DINOv2 — JAX/Flax Implementation

A from-scratch implementation of **DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)) in JAX/Flax, including all core components of the training recipe.

---

## What is DINOv2?

DINOv2 is a self-supervised vision transformer that learns powerful general-purpose visual features without any labels. It combines a self-distillation objective (DINO loss) with a masked image modeling objective (iBOT loss), producing features that transfer remarkably well to downstream tasks via simple linear probing.

---

## Implemented Components

| Component | Description |
|---|---|
| **Multi-crop augmentation** | Global + local crop strategy for student/teacher inputs |
| **EMA teacher network** | Exponential moving average of student weights, no gradient |
| **Sinkhorn-Knopp centering** | Distributed centering via SK iterations on teacher softmax |
| **iBOT patch-level loss** | Masked patch token prediction as a second SSL objective |
| **KoLeo regularization** | Uniform spread of embeddings in representation space |
| **Register tokens** | Artifact-reducing register tokens appended to ViT sequence |
| **Linear probe evaluation** | Frozen backbone + linear classifier for downstream assessment |

**Architecture:** ViT-S (Small)

---

## Implementation Notes

Non-trivial details reproduced faithfully from the paper:

- Sinkhorn-Knopp is applied only on teacher logits, with centering maintained via EMA separately from the teacher network weights
- KoLeo loss is computed on L2-normalized CLS tokens before the projection head
- Register tokens are appended after patch tokens and excluded from the iBOT loss computation
- Multi-crop uses 2 global crops (224×224) + N local crops (96×96); teacher sees only global crops

---

## Project Structure
```
dino/
|── augmentation.py          # Multi-crop augmentation pipeline
|── dino.py                  # DINO + iBOT projection heads and Sinkhorn-Knopp + cross-entropy
|── linear_eval.py           # Linear evaluation
|── train.py                 # Training loop with EMA teacher update
|── utils.py                 # KoLeo regularization
|── visualize_attention.py   # Visualize attention maps
|── vit.py                   # ViT-S architecture with register tokens
```

---

## References
```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={TMLR},
  year={2023}
}
```
