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
