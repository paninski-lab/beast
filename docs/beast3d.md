# BEAST3D

Documentation for BEAST3D multi-view self-supervised pretraining.

## Guides

- [Extracting data for BEAST3D](extract_data_for_3d.md)

---

## Environment setup

BEAST3D depends on a [custom fork of gsplat](https://github.com/QitaoZhao/gsplat) for
differentiable Gaussian splatting with intrinsics gradient support. This fork is compiled
from source during installation because it is not available as a pre-built wheel.

### Prerequisites

gsplat's `setup.py` imports `torch` at build time to find CUDA paths, so `torch` must be
installed before gsplat. The build backend (`poetry-core`) must also be present. The
standard beast install handles this:

```bash
pip install lightning poetry-core
pip install -e . --no-build-isolation
```

`--no-build-isolation` tells pip to use the existing environment (where torch is already
installed) rather than creating a temporary venv without it.

### GPU architecture

gsplat's CUDA kernels are compiled for the GPU architectures visible during the build.
The compiled kernels only run on matching hardware. If you need to target a specific
architecture (e.g. building on a machine without a GPU, or targeting a different GPU than
the one present), set `TORCH_CUDA_ARCH_LIST`:

```bash
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation
```

Common values: `7.5` (RTX 2080), `8.0` (A100), `8.6` (A40, RTX 3090), `8.9` (L40, RTX
4090), `12.0` (RTX 5090).

### Verifying the installation

```python
from gsplat.cuda._backend import _C
print(_C)   # should be a compiled module, not None
```

If the output is `None`, gsplat's CUDA extension did not load. Common causes:

- **torch was not available during the build** — install lightning first, then reinstall
  with `--no-build-isolation`.
- **CUDA toolkit not found** — ensure `CUDA_HOME` is set, or that `nvcc` is on your PATH.
- **Incompatible gcc** — nvcc requires gcc <= 12 for CUDA 12.x. Check with
  `nvcc --version` and `gcc --version`.
- **Architecture mismatch** — the kernels were compiled for a different GPU. Rebuild with
  the correct `TORCH_CUDA_ARCH_LIST`.

### Source build on Blackwell GPUs (SM 12.0) or PyTorch >= 2.7

Building on newer hardware may require additional CUDA development headers not included in
a standard conda/pip PyTorch install. These must be installed right after creating the conda
env, before running `pip install`:

```bash
conda create --yes --name beast python=3.10
conda activate beast

# install CUDA dev packages before anything else
conda install -c nvidia \
    cuda-nvcc=<cuda_version> \
    cuda-cudart-dev=<cuda_version> \
    libcusparse-dev \
    libcublas-dev \
    libcurand-dev \
    libcufft-dev \
    libcusolver-dev
```

Match `<cuda_version>` to the CUDA version you plan to use with PyTorch (e.g. `13.0`).

If conda installs CUDA headers under `targets/x86_64-linux/include/` rather than the
standard `include/`, set these environment variables before installing beast:

```bash
conda env config vars set -n beast \
    CPATH=$CONDA_PREFIX/targets/x86_64-linux/include \
    CUDA_HOME=$CONDA_PREFIX
conda activate beast   # re-activate so the variables take effect
```

Then proceed with the standard install:

```bash
pip install lightning poetry-core
pip install -e . --no-build-isolation
```
