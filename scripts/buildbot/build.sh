#! /bin/bash
# Setup a SIGINT handler. Not sure why, but this is necessary for SIGINT (Ctrl-C) to cancel this script.
handle_sigint() {
    echo "Caught SIGINT (Ctrl+C). Exiting..."
    exit 130  # Exit with a specific code (128 + signal number)
}
# Trap the SIGINT signal and call the handle_sigint function
trap handle_sigint SIGINT

set -e

USER=paninski-lab
REPO_NAME=beast

BASE_DIR=/local/$(whoami)/builds
TARGET_DIR=$BASE_DIR/$(date '+%Y_%m_%d-%H_%M_%S')
CONDA_ENV=beast_build

PR_NUMBER="${1:-0}"

echo "Running from $(hostname)"

# Activate environment
echo "Setting up environment..."
source ~/.bashrc
ml Miniforge-24.7.1-2
ml gcc/10.4                        # nvcc requires gcc <= 12 for CUDA 12.4
ml cuda/12.4.0
export CC="$NVCC_CCBIN"
export CXX="$(dirname "$NVCC_CCBIN")/g++"
export LD_PRELOAD=/home/$(whoami)/.conda/envs/$CONDA_ENV/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/share/apps/spack/gcc/14.1/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV
echo "Active conda environment: $CONDA_ENV"
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"
echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"
echo "NVCC_CCBIN=$NVCC_CCBIN"
echo "nvcc: $(which nvcc)"
echo "gcc: $(which gcc)"
$CC --version 2>&1 | head -1

# Remove builds older than 24 hours
find "$BASE_DIR" -maxdepth 1 -type d -mtime +0 -print0 | while IFS= read -r -d $'\0' directory; do
  # Skip the starting directory itself.
  if [[ "$directory" != "$BASE_DIR" ]]; then
      echo "Removing directory: $directory"
      rm -rf "$directory"
  fi
done

# Get the code of the PR.
# For efficiency, rather than cloning, it inits a blank repo
# and fetches just the code we need.
git init "$TARGET_DIR"
cd "$TARGET_DIR"
git remote add upstream "https://github.com/$USER/$REPO_NAME.git"
if [ "$PR_NUMBER" -eq 0 ]; then
  echo "No PR number provided; checking out main."
  git fetch upstream main
  git checkout FETCH_HEAD
else
  git fetch upstream "refs/pull/$PR_NUMBER/merge"
  git checkout FETCH_HEAD
fi

# Install with checks
pip install -e ".[dev]" --no-build-isolation 2>&1 | tee /tmp/pip-install.log
PIP_EXIT=${PIPESTATUS[0]}
echo "Pip install exit code: $PIP_EXIT"
if [ "$PIP_EXIT" -ne 0 ]; then
  echo "=== COMPILATION ERRORS ==="
  grep -E '(^error:|nvcc error|fatal error|cannot find|undefined reference|exit code)' /tmp/pip-install.log || echo "No specific error pattern found"
  echo "=== LAST 50 LINES ==="
  tail -50 /tmp/pip-install.log
  exit 1
fi
pip show beast-backbones
python -c "import beast; print('Beast location:', beast.__file__); print('Beast import successful')"

# Run with html reporting (for github actions) and codecov reporting
# Note the --basetemp, this puts the temporary pytest files/directories in an Axon path that is cleared daily
mkdir -p -m 777 /local/$(whoami)/pytest-tmp
pytest --html=report.html --self-contained-html --cov=. --cov-report=xml:$HOME/buildbot_lp/coverage.xml --basetemp="/local/$(whoami)/pytest-tmp/$(date '+%Y_%m_%d-%H_%M_%S')" tests/
