# BEAST3D

Documentation for BEAST3D multi-view self-supervised pretraining.

## Guides

- [Extracting data for BEAST3D](extract_data_for_3d.md)

---

## Environment setup

BEAST3D training uses [gsplat](https://github.com/nerfstudio-project/gsplat) for differentiable
Gaussian splatting. Whether you need any extra setup depends on your hardware and PyTorch
version — check first, then follow only the path that applies.

### Check whether gsplat is working

```python
from gsplat.cuda._backend import _C
print(_C)   # should be a compiled module, not None
```

If the output is a compiled module object, nothing more is needed.

If the output is `None`, gsplat's CUDA extension did not load. The fix depends on your
hardware:

- **Mainstream hardware** (A100, A6000, L40, RTX 3090/4090 — SM 8.0–8.9) with a standard
  PyTorch install (2.4–2.6, CUDA 12.1–12.4): the pre-compiled wheel should have worked.
  Check that gsplat installed correctly (`pip show gsplat`) and that your GPU is detected
  (`python -c "import torch; print(torch.cuda.is_available())"`).
- **Blackwell GPUs** (RTX 5090, SM 12.0) or **PyTorch ≥ 2.7**: no pre-compiled wheel
  supports this combination yet. You need to build gsplat from source — follow the steps
  below.

### Source build (Blackwell / PyTorch ≥ 2.7)

Building from source requires the CUDA compiler and a full set of CUDA development headers.
A standard conda/pip PyTorch install does not include these, so they must be added separately.

**Step 1 — Install CUDA build tools**

Match `cuda-nvcc` and `cuda-cudart-dev` to the CUDA version PyTorch was built against
(`python -c "import torch; print(torch.version.cuda)"`). The math-library packages can be
left unpinned:

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

**Step 2 — Persist build environment variables**

conda installs the CUDA headers under `targets/x86_64-linux/include/` rather than the
standard `include/` that the C++ host compiler searches by default. These two variables
tell the build system where to find them:

```bash
conda env config vars set -n beast \
    CPATH=/home/mattw/miniconda3/envs/beast/targets/x86_64-linux/include \
    CUDA_HOME=/home/mattw/miniconda3/envs/beast
conda activate beast   # re-activate so the variables take effect
```

**Step 3 — Clone and build gsplat**

gsplat 1.5.3 (the latest release) does not support SM 12.0 or PyTorch ≥ 2.7. The `main`
branch has both fixes. Clone it and build in-place with `--no-build-isolation`, which makes
pip use the beast env's torch headers instead of a temporary copy that won't have the CUDA
dev packages:

```bash
pip uninstall gsplat -y
git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat-src
cd /tmp/gsplat-src
git submodule update --init --depth 1
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation .
```

Compilation takes 5–15 minutes. Verify:

```bash
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA:', _C)"
```
