# SSL Methods

Seven objectives, one interface. Every one wraps the **real YOLO backbone**, so the trained
encoder transfers straight into a detector.

```python
from ssldet.ssl import available_ssl_modules
print(available_ssl_modules())
# ('byol', 'dinov2', 'dinov3', 'ijepa', 'mae', 'moco', 'simclr')
```

## Choosing one

| Method | Type | Views | Moving target | Pick it when |
|---|---|---:|---|---|
| `simclr` | Contrastive | 2 | No | You have a large per-GPU batch and want a simple, well-understood baseline |
| `byol` | Non-contrastive | 2 | EMA | Batch is small — no negatives needed |
| `moco` | Contrastive | 2 | EMA | Batch is small but you still want many negatives (queue) |
| `dinov2` | Self-distillation | 2 + local | EMA | You want local-to-global reasoning and can afford multi-crop |
| `dinov3` | Foundation distillation | 1 | Frozen | You have official DINOv3 weights and want their features |
| `mae` | Generative | 1 | No | Cheapest per step; single view, no pair construction |
| `ijepa` | Predictive | 1 | EMA | You want semantic targets without pixel reconstruction |

**Compute rule of thumb.** Single-view methods (`mae`, `ijepa`, `dinov3`) cost one encoder pass
per image. Two-view methods cost two. `dinov2` costs `2 + local_crops` student passes.

## SimCLR

Two augmented views; NT-Xent pulls the pair together against every other image in the batch.

$\ell_{i,j}=-\log\frac{\exp(s_{i,j}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)}$

Negatives come from the batch, so SimCLR is the most batch-size-sensitive method here. Negatives
are **not** gathered across GPUs — each process computes its own loss.

Key parameters: `temperature`, `projection_dim`, `hidden_dim`.

## BYOL

Online network (encoder → projector → predictor) predicts an EMA target network's output. No
negatives. Loss is symmetrized negative cosine similarity, $2-2\cos(p,z)$.

The predictor plus stop-gradient is what prevents collapse. `momentum` anneals from its start
value toward `final_momentum` on a cosine schedule.

Key parameters: `momentum`, `final_momentum`, `projection_dim`, `hidden_dim`.

## MoCo

Decouples negatives from batch size using a queue of previously encoded keys filled by an EMA key
encoder. Keys are enqueued once per optimizer step, so a gradient-accumulation group contributes
one combined batch. Under DDP, keys are all-gathered so every rank holds an identical queue.

Key parameters: `queue_size`, `temperature`, `momentum`.

## DINOv2 — compute-scaled

Multi-crop self-distillation. The student sees all crops; the EMA teacher sees only the two
global crops. Cross-entropy runs from each teacher view to every *different* student view.
Centering (running mean subtracted from teacher logits) plus sharpening (lower teacher
temperature) prevents collapse. A KoLeo term spreads features apart.

> **Adaptation notice.** Official DINOv2 is ViT-specific and also uses iBOT patch masking. This is
> a **YOLO-native compute-scaled DINOv2-style** adaptation. Do not describe it as a reproduction.

Key parameters: `dino_output_dim`, `student_temperature`, `teacher_temperature`,
`center_momentum`, `koleo_weight`, `local_crops`, `local_crop_size`.

Constraint: `0 < teacher_temperature < student_temperature`. That gap is what sharpens the target.

## DINOv3-guided distillation

Not self-distillation — a **frozen, pretrained DINOv3 teacher** supervises the YOLO student via
two cosine-regression terms:

- **global** — pooled student embedding vs teacher CLS token
- **dense** — student feature map vs teacher patch-token grid, bilinearly resized to match

Requires PyTorch 2.7.1+ and user-supplied official Meta weights. `image_size` must be divisible by
the teacher patch size (16 for ViT/16, 32 for ConvNeXt).

The teacher is excluded from saved checkpoints and re-attached on load, so checkpoints stay small.

Key parameters: `dinov3_model`, `dinov3_weights`, `dinov3_global_weight`, `dinov3_dense_weight`.

## MAE — compute-scaled

Masks a fraction of the patch grid, encodes the visible pixels, and reconstructs. Loss is MSE
over **masked pixels only** — scoring visible pixels would let the model win by copying input.

> **Adaptation notice.** This is a CNN-compatible adaptation with a lightweight conv decoder, not
> the original ViT MAE architecture.

Key parameter: `mask_ratio` (default `0.60`).

## I-JEPA — compute-scaled

Predicts in **latent space**, not pixel space. Rectangular target blocks are sampled from the
feature grid; the context encoder sees the image with those regions masked; a small transformer
predictor reconstructs the *representations* of the missing blocks against an EMA target encoder.

The predictor substitutes a shared mask token at target positions and *then* adds a sinusoidal
2-D position embedding to every token. Order matters: adding position first would let the
substitution discard it exactly where it is needed, and because self-attention is
permutation-equivariant every masked position would receive an identical prediction.

> **Adaptation notice.** Official I-JEPA uses ViT-H/14 on 16 A100s. Describe this as
> **YOLO-native compute-scaled I-JEPA**.

Key parameters: `num_target_blocks`, `target_scale_min/max`, `target_aspect_min/max`,
`predictor_depth`, `predictor_heads`, `projection_dim`.

Constraints: `projection_dim` must be divisible by 4 **and** by `predictor_heads`.

## Using an objective standalone

Any objective works with any compatible encoder — no YOLO required:

```python
from ssldet.ssl import create_ssl_module

byol = create_ssl_module(
    "byol", encoder=my_encoder,
    feature_dim=768, hidden_dim=1024, projection_dim=256, momentum=0.996,
)
loss = byol((view_a, view_b))
loss.backward()
byol.after_optimizer_step()      # EMA update; call after optimizer.step()
```

Pooled objectives need an encoder returning `B x C`. `mae`, `ijepa`, and `dinov3` additionally
need `forward_feature_map()` returning `B x C x H x W`.

To add your own, see [Contributing](Contributing.md#adding-an-ssl-objective).

## Next

[Configuration](Configuration.md) · [Model Families](Model-Families.md)
