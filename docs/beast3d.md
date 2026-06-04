# BEAST3D

Documentation for BEAST3D multi-view self-supervised pretraining.

## Guides

- [Extracting data for BEAST3D](extract_data_for_3d.md)

---

## Environment setup

BEAST3D training uses [gsplat](https://github.com/nerfstudio-project/gsplat) for differentiable
Gaussian splatting. On mainstream hardware with a supported PyTorch/CUDA combination, gsplat
installs from a pre-compiled wheel. On leading-edge hardware (e.g. RTX 5090, SM 12.0) or with
very recent PyTorch versions (≥ 2.7), a source build is required.

Either way, gsplat needs the CUDA compiler (`nvcc`) to be present — not just the GPU driver. If
`nvcc` is missing, gsplat silently disables itself and any rendering call will fail with an
`AttributeError`.

### Check whether gsplat is working

```python
from gsplat.cuda._backend import _C
print(_C)   # should be a compiled module, not None
```

If the output is `None`, follow the steps below.

### Install the CUDA compiler and library headers

Install `cuda-nvcc`, the CUDA runtime headers, and the math-library development headers into
the beast conda environment. PyTorch's ATen headers reference `cusparse.h`, `cublas.h`, and
friends, so all of those packages are required for a source build.

Match the `cuda-nvcc` and `cuda-cudart-dev` versions to the CUDA version PyTorch was built
against (check with `python -c "import torch; print(torch.version.cuda)"`). The math-library
packages (`libcusparse-dev`, etc.) can be left unpinned and conda will resolve compatible
versions automatically.

```bash
conda activate beast
conda install -c nvidia \
    cuda-nvcc=<cuda_version> \
    cuda-cudart-dev=<cuda_version> \
    libcusparse-dev \
    libcublas-dev \
    libcurand-dev \
    libcufft-dev \
    libcusolver-dev
```

For example, with CUDA 13.0:

```bash
conda install -c nvidia \
    cuda-nvcc=13.0 \
    cuda-cudart-dev=13.0 \
    libcusparse-dev \
    libcublas-dev \
    libcurand-dev \
    libcufft-dev \
    libcusolver-dev
```

Persist the two environment variables that point the build system at the right paths.
These need to be set before any source compilation and will be active automatically whenever
the `beast` environment is activated:

```bash
conda env config vars set -n beast \
    CPATH=/home/mattw/miniconda3/envs/beast/targets/x86_64-linux/include \
    CUDA_HOME=/home/mattw/miniconda3/envs/beast
```

Re-activate the environment so the variables take effect:

```bash
conda activate beast
```

### Source build (for leading-edge hardware or recent PyTorch)

JIT compilation (the default path) can fail for two reasons:

- **PyTorch API mismatch** — gsplat 1.5.3 calls an internal PyTorch JIT helper
  (`_jit_compile`) that gained new required arguments in PyTorch ≥ 2.11. gsplat's
  `_backend.py` partially handles this but the fallback path may still fail to compile.
- **Unsupported GPU architecture** — gsplat 1.5.3 does not have Blackwell-compatible CUDA
  kernel code for SM 12.0 targets.

Both issues are fixed on the gsplat `main` branch (commit `355ddaf`, PR #651). Clone and
build from source:

```bash
pip uninstall gsplat -y
git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat-src
cd /tmp/gsplat-src
git submodule update --init --depth 1
```

Then build with `--no-build-isolation` so the build uses the beast env's torch headers
directly rather than a temporary pip-managed copy that may be missing the CUDA dev packages:

```bash
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation .
```

The compilation takes 5–15 minutes. Verify afterwards:

```bash
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA:', _C)"
```

### Notes on mainstream hardware

On mainstream hardware (A100, A6000, L40, RTX 3090/4090 — SM 8.0 through 8.9) with a
standard PyTorch install (2.4–2.6, CUDA 12.1–12.4), `pip install gsplat` installs a
pre-compiled wheel with no compilation step required.

The source-build steps above are only needed on NVIDIA's Blackwell generation (RTX 5090,
SM 12.0) or with very recent PyTorch versions (≥ 2.7), where gsplat 1.5.3 pre-compiled
wheels are not yet available.
