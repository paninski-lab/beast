# Buildbot CI Scripts

These scripts run the test suite on the Axon SLURM cluster. The GitHub Actions
workflow (`.github/workflows/tests.yaml`) SSHs into Axon and calls
`build_srun.sh`, which submits `build.sh` as a SLURM job.

```
.github/workflows/tests.yaml
  → ssh axon 'sh buildbot/build_srun.sh <PR_NUMBER>'
    → srun ... buildbot/build.sh <PR_NUMBER>
```

## How it works

### `build_srun.sh`

Submits `build.sh` via `srun` with resource requests:

- **GPU type: `a40`** — gsplat's CUDA kernels are compiled for a specific GPU
  architecture (sm_86 for A40). The job must run on the same GPU type the kernels
  were built for, otherwise you get `no kernel image is available for execution
  on the device` errors at runtime. See "gsplat and GPU architecture" below.
- **Memory: 32GB** — the test suite loads large models (ViT ~450MB, DINOv2, etc.)
  plus DataLoader workers. 8GB is not enough and causes OOM kills.
- **`-X`** — forwards SIGINT so GitHub Actions cancellation works.
- **`-u`** — unbuffered output for real-time log streaming.

### `build.sh`

1. **Environment setup** — loads modules (gcc, CUDA), activates the conda env,
   and sets compiler variables.
2. **Fetches PR code** — shallow-clones just the PR merge ref.
3. **Installs beast** — `pip install -e ".[dev]" --no-build-isolation`.
4. **Runs pytest** — with coverage reporting for Codecov.

## gsplat and GPU architecture

`beast` depends on a [custom gsplat fork](https://github.com/QitaoZhao/gsplat)
that compiles CUDA kernels from source. These kernels are compiled for a specific
GPU architecture (e.g. sm_86 for A40) and will only run on matching hardware.

Key constraints:

- **gsplat requires `torch` at build time** — its `setup.py` imports torch to
  find CUDA paths. This is why `--no-build-isolation` is needed (so pip uses the
  existing torch instead of creating a clean venv without it).
- **gsplat requires a CUDA-compatible gcc** — CUDA 12.4 requires gcc <= 12.
  The `gcc/10.4` module provides gcc 12.3 (the module name doesn't match the
  actual version). The `cuda/12.4.0` module sets `NVCC_CCBIN` to this compiler,
  and `build.sh` exports it as `CC`/`CXX` so torch's build system passes it
  to nvcc.
- **gsplat requires compute capability >= 7.0** — it uses
  `cg::labeled_partition` from CUDA cooperative groups, which is not available
  on Pascal GPUs (sm_61, e.g. GTX 1080). This rules out ax[03-05].
- **The compiled kernels are cached in the conda env** — gsplat takes ~30
  minutes to build. Once installed, it persists across CI runs as long as the
  conda env and GPU type stay the same. The `srun` GPU type constraint
  (`--gres=gpu:a40:2`) ensures the job always lands on a matching node.

### Checking the installed gsplat architecture

```bash
/share/apps/cuda/12.4/bin/cuobjdump -lelf \
  $(python -c "import gsplat.csrc as m; print(m.__file__)") | grep sm_
```

### Rebuilding gsplat for a different GPU type

If you need to switch GPU types (e.g. from A40 to L40), you must rebuild gsplat
on a node with the target GPU. Run interactively:

```bash
srun --gres=gpu:a40:1 --mem=16g -c4 -t 1:00:00 bash -c '
  ml gcc/10.4 cuda/12.4.0
  export CC="$NVCC_CCBIN"
  export CXX="$(dirname "$NVCC_CCBIN")/g++"
  conda activate beast_build
  pip uninstall -y gsplat
  pip install --no-cache-dir \
    "gsplat @ git+https://github.com/QitaoZhao/gsplat.git@daad91d9e667cf49bab815d872aa65a5cda0a77e" \
    --no-build-isolation
'
```

Then update `--gres=gpu:<type>:2` in `build_srun.sh` to match.

**Important:** use `--no-cache-dir` when rebuilding — pip caches wheels by
package version, so without this flag it will reuse the old wheel compiled for
the wrong architecture.

## Compiler toolchain

The build environment juggles two gcc versions:

| Purpose | Version | Source |
|---|---|---|
| CUDA host compiler (nvcc `-ccbin`) | gcc 12.3 | `ml gcc/10.4` + `ml cuda/12.4.0` sets `NVCC_CCBIN` |
| Runtime `libstdc++` (NumPy 2 requirement) | gcc 14.1 | `LD_PRELOAD` + `LD_LIBRARY_PATH` in `build.sh` |

The `gcc/10.4` module name is misleading — it provides gcc 12.3 via spack. The
CUDA module depends on it and sets `NVCC_CCBIN` to the correct binary path.
`build.sh` then exports `CC` and `CXX` from `NVCC_CCBIN` so that torch's
`cpp_extension.py` passes the right compiler to nvcc.

The gcc 14.1 `libstdc++` is only needed at runtime (via `LD_PRELOAD`) for NumPy
2 compatibility — it is not used as a compiler.

## Initial setup

### I. Self-hosted runner

A self-hosted GitHub Actions runner is required with SSH access to Axon.

1. You need a server (e.g. AWS, a physical PC). Axon nodes do not support
   outbound internet so they cannot serve as runners directly.

2. Configure SSH access to Axon. Ensure `~/.ssh/config` contains:
   ```
   Host axon
       HostName axon-remote.rc.zi.columbia.edu
       User <YOUR_UNI>
       Port 55
   ```

3. Set up passwordless SSH authentication via `ssh-keygen` and `ssh-copy-id`.

4. Install the GitHub self-hosted runner software. Follow the steps at:
   Repository settings > Actions > Runners > New self-hosted runner.

### II. Axon setup

Create the buildbot directory and scripts:

```
~/buildbot/build.sh
~/buildbot/build_srun.sh
```

Make `build.sh` executable:

```bash
chmod +x ~/buildbot/build.sh
```

Set up the conda environment:

```bash
conda create -n beast_build python=3.11
conda activate beast_build
conda install -c conda-forge opencv   # needed for ffmpeg
pip install lightning poetry-core
```

Then install gsplat (see "Rebuilding gsplat" above) and beast:

```bash
pip install -e ".[dev]" --no-build-isolation
```

### III. Test the setup

From the self-hosted runner server:

```bash
ssh axon 'sh buildbot/build_srun.sh <PR_NUMBER>'
```

### IV. Security

Ensure external contributors cannot run arbitrary workflows on your self-hosted
runner — they would be able to execute code on Axon with your UNI's permissions.

Configure this in GitHub:

Settings > Actions > General > Approval for running fork pull request workflows

The default is "Require approval for first-time contributors". For maximum
security, restrict to repository owners or organization members.
