# Roadmap

Known issues and planned work that are not yet scheduled.

## Support newer `transformers` releases

`transformers` is capped at `<5.9` in `pyproject.toml`. The ViT-MAE model in
`beast/models/beast_vit/beast_vit_model.py` is built on `ViTMAEForPreTraining`, and newer releases
(checked against 5.22.1 by reading the source; not yet run) change it in ways that break beast:

- **Removed helper.** `get_2d_sincos_pos_embed` is no longer in
  `transformers.models.vit_mae.modeling_vit_mae`. It was replaced by
  `build_2d_sinusoidal_position_embedding(height, width, embed_dim, cls_token)`, which returns a
  torch tensor. Its `initialize_weights` swaps the two halves of the result so that the layout
  matches the original MAE (pretrained) weights. This broke pyright in CI, which installs the
  latest release.
- **Restructured encoder.** `ViTMAEEncoder` is gone and `ViTMAEModel` holds `self.layers`
  directly. Checkpoint keys change from `vit.encoder.layer.N.*` to `vit.layers.N.*`, so:
  - `beast/api/model.py` (`model.load_state_dict(...)`, strict) would fail on checkpoints saved
    with older versions.
  - `ViTMAE.forward` calls `self.vit.encoder(...)` in the unmasked path (`mask_ratio == 0`,
    used for inference and latents), which would raise an `AttributeError`.
- **Random init.** With 5.8.0, building a model from a config (`random_init: true`) leaves the
  frozen position embeddings, class token and mask token at zero;
  `init_mae_weights` works around this. Whether newer releases still do this is untested.

Possible fixes:

1. Replace the explicit sin-cos computation in `init_mae_weights` with the model's own
   `embeddings.initialize_weights()` and `decoder.initialize_weights(num_patches)`, which exist
   in both versions (clear the private `_is_hf_initialized` flag on the patch projection first,
   since it makes the first one return early). No transformers code is copied or imported.
2. Alternatively, import the helper in a `try` / `except ImportError` block, trying
   `build_2d_sinusoidal_position_embedding` (and rotating the two halves) before falling back to
   `get_2d_sincos_pos_embed`.
3. Remap checkpoint keys on load (`vit.encoder.layer.` <-> `vit.layers.`) in
   `beast/api/model.py`, and handle both layouts in `ViTMAE.forward`.
4. Add a CI job that runs the test suite against the oldest and newest supported `transformers`
   so this is caught before release, and find the exact release where the changes land.

## Distill a small MAE (ViT-S) from a larger one (DMAE)

Plain MAE pretraining is weak for small ViTs: in the TinyMIM paper it made ViT-T worse than no
pretraining and helped ViT-S by only 0.7 points, while distillation from a MAE teacher gave
+2.4 on ViT-S. Plan: add feature distillation from a frozen, already-trained ViT-B MAE into the
ViT-S run, following DMAE ([Bai et al.](https://arxiv.org/pdf/2208.12256)).

**Method**

- The student trains as usual (same mask, reconstruction loss on masked patches) and adds
  `L = L_mae + alpha * L_dist`, with `alpha = 1`.
- The teacher is frozen and sees the same visible patches as the student. It only needs to run
  up to the aligned layer, so its remaining layers can be skipped.
- Align one layer at 3/4 of the depth in both networks (matched by relative position). For a
  12-layer ViT-B teacher and 12-layer ViT-S student that is block 9 to block 9. Aligning
  several layers was no better (84.2 vs 84.0 in the paper's ViT-B ablation); shallow layers
  were worst.
- `L_dist` is the L1 distance between the teacher's block output and
  `proj(student block output)` over the visible patch tokens only (class token excluded or
  handled consistently on both sides).
- `proj` is a 2-layer MLP with GELU, hidden and output size equal to the teacher dimension
  (384 -> 768 for ViT-S from ViT-B). It is only used during pretraining and is dropped
  afterwards.
- Optimizer, lr, wd, and betas follow the existing MAE recipe. The paper trains for 100 epochs
  and shows it still works at mask ratios up to 90% with no loss in accuracy, so a shorter
  schedule may be enough; verify on our data.

**Implementation sketch**

- Config: add a `distillation` section to the ViT config schema with `teacher_checkpoint`,
  `layer_frac` (default 0.75), and `weight` (default 1.0); add it to the shipped yamls and a
  schema test.
- Model: load the teacher once, set it to `eval()` with `requires_grad_(False)`, and run it
  under `torch.no_grad()` with the student's `noise` or `ids_keep` so both see identical
  visible patches. Read the hidden state at the chosen block (e.g. `output_hidden_states`).
- Log `train_loss_recon` and `train_loss_distill` separately.
- Tests: teacher gets no gradients, teacher and student see the same patches, loss shapes
  match for mismatched dims, and the config round trips.

**Open questions**

- How the class token and the final layernorm are handled in the HF `ViTMAE` hidden states.
- Whether to also try the variants below if DMAE falls short.

**Alternatives if DMAE is not enough**

- MAE-Lite ([Wang et al.](https://arxiv.org/pdf/2205.14443)): match attention maps of the last
  layers, plus reconstruction. Needs attention weights (likely forces eager attention instead
  of fused kernels), and head counts differ between ViT-B and ViT-S.
- G2SD ([Huang et al.](https://arxiv.org/pdf/2302.14771)): also aligns the teacher decoder's
  hidden features at masked positions, then a task-specific second stage.
- TinyMIM ([Ren et al.](https://arxiv.org/pdf/2301.01296)): token-relation targets; best
  reported ViT-S numbers (83.0 top-1) and a ViT-B -> ViT-S -> ViT-T chain beat direct
  distillation.
- Asymmetric Masked Distillation ([Zhao et al.](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Asymmetric_Masked_Distillation_for_Pre-Training_Small_Foundation_Models_CVPR_2024_paper.pdf)):
  teacher and student get different masks.
- NoR-MAE ([Meng et al.](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Closest_Neighbors_are_Harmful_for_Lightweight_Masked_Auto-encoders_CVPR_2025_paper.pdf)):
  claims a strong ViT-Tiny with no teacher by changing the MAE objective; read in full before
  relying on it.

No controlled head-to-head of these methods under one teacher and schedule was found, and the
numbers above come from each paper's own setup. The literature search covered work up to
2025; newer JEPA/DINO-style recipes were not checked.

## Higher-resolution continuation (experimental)

Idea: pretrain at the current resolution (224), then continue for a short schedule at 384 x 384
(upsampled images), the way DINOv3 ends training with a high-resolution phase. This is more
speculative than the distillation item above: no published MAE work was found that does a
low-res stage followed by a short high-res stage, so any benefit would be new evidence from our
own data.

**What is known**

- DINOv3 ([report](https://www.emergentmind.com/papers/2508.10104)) adds a short
  high-resolution adaptation phase (about 10k iterations per a secondary source, unconfirmed)
  after the main run, with Gram anchoring. Its main run uses a constant lr schedule so a branch
  can start at any point. Gram anchoring has no direct MAE analogue.
- DailyMAE ([arXiv 2404.00509](https://arxiv.org/pdf/2404.00509)) varies resolution during MAE
  pretraining (160-224) for speed and finds that decreasing then increasing resolution (a
  "palindrome" schedule) works better than monotonically increasing it. It does not test
  whether ending at a larger resolution helps.
- ViTDet-style work pretrains MAE at 224 and fine-tunes at 512+; the main reported risk is
  position-embedding handling
  ([arXiv 2311.05613](https://ar5iv.labs.arxiv.org/html/2311.05613)).

**Considerations**

- Upsampling adds no new image information. Any gain would come from finer patch granularity
  (a 16 px patch covers less of the scene), not from extra detail. 224 -> 384 goes from 14 x 14
  = 196 to 24 x 24 = 576 tokens (about 2.9x); the encoder only sees the visible 25%, so the
  cost is lower than a plain ViT at the same size.
- Keep the patch size fixed so the grid grows. The sin-cos position embeddings are fixed and
  can be regenerated for the new grid; check how the HF `ViTMAE` here handles a different
  `image_size` (model config, `interpolate_pos_encoding`, and `init_mae_weights`).
- Use a short schedule with a lower lr (e.g. resume near the end-of-schedule lr, not the peak)
  so the low-res features are not wrecked.
- Check that the viz sheets, `num_patches`, and mask ratio bookkeeping all read the image size
  from the config.

**Experiment**

Continue the same checkpoint for N steps under three settings: (a) 384 upsampled, (b) 224, same
extra steps and lr, (c) 384 with the 224 position embeddings interpolated. Compare val
reconstruction loss at matched resolution and, more importantly, the downstream task at its
native input size. (b) separates "more training" from "higher resolution".
