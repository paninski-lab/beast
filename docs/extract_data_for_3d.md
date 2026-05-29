# Extracting Data for BEAST3D

This guide walks through preparing a multi-view video dataset for BEAST3D training.
The pipeline takes raw synchronized multi-view videos and camera calibration files,
and produces a structured dataset of frames, camera parameters, and (optionally)
segmentation masks.

---

## Overview

The pipeline runs up to six steps in sequence, each gated by a flag in the config:

| Step | Config flag | What it does |
|------|-------------|--------------|
| Stats | always runs | Scans input videos; writes `video_stats.csv` and `video_stats.json` |
| Trim | `trim.enabled` | Clips every video to a fixed frame/time range |
| Downsample | `downsample.enabled` | Re-encodes videos at a lower frame rate or frame count |
| Segment | `segmentation.enabled` | Runs SAM3 video tracking to produce per-frame binary masks |
| Assemble | always runs | Selects frames via k-means, exports images and camera params |
| Resize | `resize.enabled` | Rescales assembled images and masks in-place |

Run the full pipeline with one command:

```bash
beast extract_3d --config configs/multiview/extraction_pipeline.yaml
```

Pass `--skip-stats` to skip the stats step if you have already run it and do not
want to re-scan the videos.

A complete, annotated config is at
[configs/multiview/extraction_pipeline.yaml](../configs/multiview/extraction_pipeline.yaml).

---

## Input Data

### Directory layout

```
input_dir/
├── videos/                        # raw video files
│   ├── session01_cam0.mp4
│   ├── session01_cam1.mp4
│   ├── session01_cam2.mp4
│   ├── session02_cam0.mp4
│   ├── session02_cam1.mp4
│   └── session02_cam2.mp4
└── calibrations/                  # one calibration file per session
    ├── session01.toml
    └── session02.toml
```

`input_dir` and `output_dir` are set in the config. Everything else (`videos/`,
`calibrations/`) lives under `input_dir` by convention.

### Video naming convention

Every video filename must match the pattern set in `video.filename_pattern`.
The default regex is:

```
^(?P<session>.+)_(?P<cam>[^_]+)\.mp4$
```

This requires exactly two named groups:

- **`session`** — groups videos that were recorded simultaneously (e.g., `session01`)
- **`cam`** — identifies the camera within a session (e.g., `cam0`, `cam1`)

A video named `session01_cam0.mp4` yields `session=session01`, `cam=cam0`.
All views sharing the same session name are treated as synchronized and will have
frames selected from the same time points (chosen by k-means on the anchor view).

If your filenames have a different structure, update `video.filename_pattern` with
a regex that still captures `session` and `cam` groups.

### Calibration files

Camera calibration is expected in **anipose TOML format**, produced by
[aniposelib](https://github.com/lambdaloop/aniposelib).
One file per session, located at `input_dir/calibrations/{session_id}.toml`.

Each TOML file stores per-camera intrinsics (3×3 matrix), extrinsics (4×4 matrix),
and distortion coefficients. The pipeline reads these and saves them alongside each
extracted frame as a `.npy` file.

If calibration files are missing for some sessions, those sessions are still
processed but frames are exported without camera parameter files.

#### Mapping camera IDs to calibration names

The `cam` group parsed from the video filename may not match the camera names stored
inside the calibration file. Use `calibration.cam_name_mapping` to bridge the gap:

| Value | Effect |
|-------|--------|
| `identity` | camera name == cam_id (default) |
| `prefix_<str>` | camera name = `<str>` + cam_id, e.g. `prefix_Cam-` → `Cam-0` |
| any string with `{cam_id}` | used as a format string, e.g. `camera_{cam_id}` |

---

## Output Structure

```
output_dir/
├── video_stats.csv              # per-video stats (fps, resolution, frame count)
├── video_stats.json             # aggregate summary stats
├── videos_trim/                 # (cut.enabled) trimmed videos
├── videos/                      # (downsample.enabled) downsampled videos
├── segmentation_masks/
│   ├── session01_cam0/
│   │   ├── masks/
│   │   │   ├── mask00000000.png
│   │   │   ├── mask00000001.png
│   │   │   └── ...
│   │   ├── tracking.json        # per-frame tracking metadata
│   │   └── _COMPLETE            # written on success; absence = incomplete run
│   └── failed_videos.json       # present only if any videos failed
└── dataset/
    ├── info.json                # dataset-level metadata
    ├── session01/
    │   ├── selected_frames.csv  # list of exported frame filenames
    │   ├── cam0/
    │   │   ├── img00000042.png  # extracted frame
    │   │   ├── img00000042.npy  # camera params dict (+ bbox if has_bboxes)
    │   │   └── mask00000042.png # segmentation mask (if segmentation.enabled)
    │   ├── cam1/
    │   │   └── ...
    │   └── cam2/
    │       └── ...
    └── session02/
        └── ...
```

The `.npy` camera parameter files contain a dict with keys `intrinsics`,
`extrinsics`, `distortions`, `width`, `height`, and optionally `bbox`.
Load with `np.load('img00000042.npy', allow_pickle=True).item()`.

---

## Configuration Reference

All settings live in a single YAML file. Only `name`, `input_dir`, `output_dir`,
and `anchor_view` are required to be non-empty; everything else has a sensible default.

### Top-level fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | `''` | Short name for this dataset (written into `info.json`) |
| `input_dir` | `''` | Root directory of raw data; must contain `videos/` and (for calibrated runs) `calibrations/` |
| `output_dir` | `''` | All pipeline outputs are written here |
| `anchor_view` | `''` | Camera whose frames drive k-means selection; must match a `cam` group in the filenames |
| `video_subdir` | `videos` | Subdirectory under `input_dir` that holds video files |
| `max_workers` | `4` | Parallel workers used by the trim, downsample, and assemble steps |
| `has_bboxes` | `false` | Set to `true` if bounding box CSVs are available alongside videos |
| `bbox_csv_pattern` | `''` | Filename template for bbox CSVs, e.g. `{session_id}_{cam_id}_bbox.csv` |
| `author` | `anonymous` | Written into `info.json` |
| `seed` | `42` | Random seed for frame sampling fallback |

### `video`

Controls video file discovery.

| Field | Default | Description |
|-------|---------|-------------|
| `filename_pattern` | `^(?P<session>.+)_(?P<cam>[^_]+)\.mp4$` | Regex with named groups `session` and `cam` |
| `extensions` | `[mp4, avi]` | File extensions to scan for |

### `frame`

Controls how frames are selected and exported in the assemble step.

| Field | Default | Description |
|-------|---------|-------------|
| `frames_per_video` | `1000` | Number of frames to select per session (k-means on anchor view) |
| `n_digits` | `8` | Zero-padding width for exported filenames, e.g. `img00000042.png` |
| `extension` | `png` | Image format for exported frames |
| `kmeans_resize` | `32` | Each frame is resized to this square size before k-means pixel clustering |

### `segmentation`

Controls the SAM3 video segmentation step. Requires a GPU.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Set to `false` to skip segmentation entirely |
| `text_prompt` | `animal` | Grounding text prompt for SAM3 object detection on the first clip |
| `num_objects` | `null` | Fix the number of tracked objects; `null` lets SAM3 decide |
| `threshold` | `0.5` | SAM3 confidence threshold for mask acceptance |
| `clip_size` | `512` | Number of frames processed per SAM3 inference clip |

The segmentation step is the most compute-intensive. Each video is processed on a
single GPU; if multiple GPUs are available, videos are distributed round-robin and
processed in parallel. The first run downloads the SAM3 model from HuggingFace
(`facebook/sam3`); subsequent runs use the local cache.

Segmentation outputs are written to `output_dir/segmentation_masks/{video_stem}/`.
A `_COMPLETE` sentinel file is written on success. If a run is interrupted, the
incomplete output directory is cleaned automatically on the next run.

### `trim`

Optional trim step. Disabled by default.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Set to `true` to trim all videos |
| `start_frame` | `null` | First frame to keep (inclusive); takes priority over `start_sec` |
| `end_frame` | `null` | Last frame to keep (inclusive); takes priority over `end_sec` |
| `start_sec` | `null` | Start time in seconds (converted to frames using source FPS) |
| `end_sec` | `null` | End time in seconds |
| `ffmpeg_threads` | `null` | Threads per ffmpeg process; `null` = `cpu_count / max_workers` |

Frame-based bounds take priority over second-based bounds. If neither is set for
a bound, it defaults to the beginning or end of the video.

Trimmed videos are written to `output_dir/{video_subdir}_trim/` (e.g., `output_dir/videos_trim/`).

### `downsample`

Optional frame-rate reduction step. Disabled by default.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Set to `true` to downsample all videos |
| `target_fps` | `null` | Target frame rate in fps |
| `ffmpeg_threads` | `null` | Threads per ffmpeg process; `null` = `cpu_count / max_workers` |
| `phase_offset_frames` | `0` | Skip this many frames at the start before downsampling begins |

For example, setting `target_fps: 1` on a 25 fps video selects every 25th frame,
reducing a 10-minute video from ~15,000 frames to ~600. This is useful when working
with long videos where the full frame pool would be redundant for k-means selection.
The assemble step then runs k-means on the downsampled frames to pick `frames_per_video`
of them, so the two steps compose: downsample narrows the candidate pool, k-means
picks the most visually diverse subset.

When both `trim` and `downsample` are enabled, downsample reads from the trim output.
Downsampled videos are written to `output_dir/{video_subdir}/` (e.g., `output_dir/videos/`).

### `calibration`

Controls how camera calibration files are located and parsed.

| Field | Default | Description |
|-------|---------|-------------|
| `format` | `anipose_toml` | Calibration file format; currently only `anipose_toml` is supported |
| `cam_name_mapping` | `identity` | How to map cam_id from filename to camera name in the calibration file; see [Mapping camera IDs to calibration names](#mapping-camera-ids-to-calibration-names) |
| `file_pattern` | `{session_id}.toml` | Filename template for calibration files; `{session_id}` is filled from the parsed `session` group |

### `assemble`

Controls the final frame export step.

| Field | Default | Description |
|-------|---------|-------------|

### `resize`

Optional in-place resize of all assembled images and masks. Disabled by default.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Set to `true` to resize after assembly |
| `size` | `256` | Target size in pixels for the **shorter** side; aspect ratio is preserved |

Camera intrinsics in the `.npy` files are updated automatically to reflect the new
image dimensions.

---

## Tips

**Running on a cluster with multiple GPUs**
Set `CUDA_VISIBLE_DEVICES` before calling `beast extract_3d`, or let the pipeline
auto-detect all available GPUs. Videos are split round-robin across GPUs and
processed in parallel.

**Skipping already-complete steps**
The stats, trim, and downsample steps overwrite outputs each run. The segmentation
step is idempotent: it skips any video whose output directory already has a
`_COMPLETE` sentinel file. To force re-segmentation of a video, delete its output
directory under `output_dir/segmentation_masks/`.

**Re-running without re-scanning videos**
Pass `--skip-stats` to skip the stats scan when re-running after an interrupted run:

```bash
beast extract_3d --config my_config.yaml --skip-stats
```

**Videos without calibration**
If a session's calibration file is missing, the pipeline logs a warning and continues.
Frames are exported without `.npy` camera parameter files. This is useful for
collecting raw training images when calibration data is not yet available.
