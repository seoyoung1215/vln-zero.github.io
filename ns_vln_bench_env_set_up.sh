#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HABITAT_SIM_TARBALL="habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2"
HABITAT_SIM_URL="https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/${HABITAT_SIM_TARBALL}"


cd "$SCRIPT_DIR"


conda_install() {
  if command -v mamba >/dev/null 2>&1; then
    mamba install -y "$@"
  else
    conda install -y "$@"
  fi
}


if [[ ! -f "$HABITAT_SIM_TARBALL" ]]; then
  echo "Downloading habitat-sim tarball..."
  wget -O "$HABITAT_SIM_TARBALL" "$HABITAT_SIM_URL"
fi


echo "Installing habitat-sim..."
conda_install "$HABITAT_SIM_TARBALL"
# mamba install -y -c aihabitat "habitat-sim=0.3.3=py3.9_headless_linux_*"


echo "Installing habitat-lab dependencies..."
pushd habitat-lab >/dev/null
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
echo "Note: if rl requirements fail, remove/comment tensorflow in habitat_baselines/rl/requirements.txt and rerun."
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python -m pip install -e .
popd >/dev/null


echo "Installing project dependencies..."
python -m pip install tokenizers==0.19.1
python -m pip install tensorboard
python -m pip install -r requirements.txt
python -m pip install "webdataset<0.2"
python -m pip install openai==1.106.1
python -m pip install "python-dotenv>=1.0.0"


echo "Installing conda dependency (python-lmdb)..."
conda_install "$HABITAT_SIM_TARBALL"
# mamba install -y -c aihabitat "habitat-sim=0.3.3=py3.9_headless_linux_*"


echo "Setup complete."
