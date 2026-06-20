#!/usr/bin/env bash
# Build an isolated RoboTwin Hy-VLA eval env that mirrors the RLinf-good
# torch/cuRobo/warp stack, without modifying RLinf's .venv.

set -euo pipefail

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-/home/jovyan/code/wge/RoboTwin_hy}
BASE_ENV=${BASE_ENV:-RoboTwinHy}
ENV_NAME=${ENV_NAME:-RoboTwinHy26}
TORCH_VERSION=${TORCH_VERSION:-2.6.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.21.0}
WARP_VERSION=${WARP_VERSION:-1.11.1}
SAPIEN_VERSION=${SAPIEN_VERSION:-3.0.1}
MPLIB_VERSION=${MPLIB_VERSION:-0.2.1}
CUROBO_REF=${CUROBO_REF:-a35a708ecfbb26eb9ab2d7ef22c65919c4fae4a9}
CUROBO_SPEC=${CUROBO_SPEC:-"nvidia-curobo @ git+https://ghfast.top/https://github.com/NVlabs/curobo.git@${CUROBO_REF}"}
FLASH_ATTN_URL=${FLASH_ATTN_URL:-"https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

has_conda_env() {
    conda env list | awk 'NF {print $1}' | grep -qx "$1"
}

restore_if_present() {
    local live=$1
    local backup=$2
    if [ -f "${backup}" ]; then
        cp "${backup}" "${live}"
        echo "[restore] ${live} <- ${backup}"
    fi
}

[ -d "${HY_VLA_DIR}" ] || die "HY_VLA_DIR not found: ${HY_VLA_DIR}"
[ -d "${ROBOTWIN_DIR}" ] || die "ROBOTWIN_DIR not found: ${ROBOTWIN_DIR}"

if [ -n "${VIRTUAL_ENV:-}" ] && [[ "${VIRTUAL_ENV}" == *"/RLinf/.venv"* ]]; then
    die "RLinf .venv is active (${VIRTUAL_ENV}). Run 'deactivate' before this script."
fi

if ! command -v conda >/dev/null 2>&1; then
    die "conda is not on PATH"
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

has_conda_env "${BASE_ENV}" || die "base env '${BASE_ENV}' not found"
if has_conda_env "${ENV_NAME}"; then
    if [ "${HYVLA_RECREATE_ENV:-0}" = "1" ]; then
        echo "[conda] removing existing env: ${ENV_NAME}"
        conda env remove -n "${ENV_NAME}" -y
    else
        die "target env '${ENV_NAME}' already exists. Set HYVLA_RECREATE_ENV=1 to rebuild it."
    fi
fi

echo "[conda] cloning ${BASE_ENV} -> ${ENV_NAME}"
conda create -n "${ENV_NAME}" --clone "${BASE_ENV}" -y
conda activate "${ENV_NAME}"

if [ -n "${VIRTUAL_ENV:-}" ] && [[ "${VIRTUAL_ENV}" == *"/RLinf/.venv"* ]]; then
    die "still inside RLinf .venv after conda activation; aborting"
fi

if [ "${HYVLA_RESTORE_ROBOTWIN_PATCHES:-1}" = "1" ]; then
    if [ -f "${ROBOTWIN_DIR}/script/eval_policy.py.hyvla_traceback.bak" ]; then
        restore_if_present \
            "${ROBOTWIN_DIR}/script/eval_policy.py" \
            "${ROBOTWIN_DIR}/script/eval_policy.py.hyvla_traceback.bak"
    else
        restore_if_present \
            "${ROBOTWIN_DIR}/script/eval_policy.py" \
            "${ROBOTWIN_DIR}/script/eval_policy.py.hyvla_skip_expert.bak"
    fi
    restore_if_present \
        "${ROBOTWIN_DIR}/envs/robot/planner.py" \
        "${ROBOTWIN_DIR}/envs/robot/planner.py.hyvla_no_graph.bak"
fi

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-jovyan}"
mkdir -p "${XDG_RUNTIME_DIR}"

echo "[pip] replacing torch/cuRobo/warp/SAPIEN stack in ${ENV_NAME}"
# SAPIEN 3.0.1 still imports pkg_resources. Newer setuptools builds may not
# provide it, so keep setuptools on the last known-compatible major line.
python -m pip install -U pip wheel "setuptools<81"
python -m pip uninstall -y torch torchvision torchaudio nvidia-curobo curobo warp-lang sapien mplib || true
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
python -m pip install \
    numpy==1.26.4 \
    numpy-quaternion==2024.0.13 \
    "sapien==${SAPIEN_VERSION}" \
    "mplib==${MPLIB_VERSION}" \
    "warp-lang==${WARP_VERSION}"
python -m pip install "${CUROBO_SPEC}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[conda] ffmpeg binary missing; installing into ${ENV_NAME}"
    conda install -c conda-forge ffmpeg -y
fi

echo "[pip] installing Hy-VLA adapter/runtime deps without pulling the training stack"
python -m pip install -e "${HY_VLA_DIR}" --no-deps
python -m pip install -U \
    "transformers>=4.57,<4.58" \
    safetensors \
    "huggingface-hub>=0.23" \
    timm==1.0.21 \
    scipy

if [ "${HYVLA_SKIP_FLASH_ATTN:-0}" != "1" ]; then
    python -m pip install "${FLASH_ATTN_URL}"
fi

if [ "${HYVLA_REBUILD_PYTORCH3D:-0}" = "1" ]; then
    echo "[pip] rebuilding pytorch3d for the new torch stack"
    python -m pip uninstall -y pytorch3d || true
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
        --no-build-isolation
fi

export HY_VLA_DIR ROBOTWIN_DIR ENV_NAME BASE_ENV
python - <<'PY'
import importlib.util
import os
import sys

def show(name, module):
    version = getattr(module, "__version__", "<unknown>")
    path = getattr(module, "__file__", "<unknown>")
    print(f"{name:14}: {version} {path}", flush=True)

print("python        :", sys.executable, flush=True)

import torch
print("torch         :", torch.__version__, torch.version.cuda, flush=True)

import warp
show("warp", warp)

import pkg_resources  # noqa: F401
print("pkg_resources: OK", flush=True)

import sapien
show("sapien", sapien)

import mplib
show("mplib", mplib)

import curobo
show("curobo", curobo)

robotwin_dir = os.environ["ROBOTWIN_DIR"]
source_curobo = os.path.abspath(os.path.join(robotwin_dir, "envs", "curobo", "src"))
curobo_file = os.path.abspath(getattr(curobo, "__file__", ""))
if curobo_file.startswith(source_curobo):
    raise SystemExit(
        "curobo still imports from RoboTwin source tree, expected site-packages: "
        + curobo_file
    )

missing = [
    name
    for name in ["transformers.modeling_layers", "timm", "flash_attn"]
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("missing Hy-VLA runtime module(s): " + ", ".join(missing))

import transformers
show("transformers", transformers)

import hy_vla  # noqa: F401
print("hy_vla        : import OK", flush=True)

try:
    import pytorch3d
except Exception as exc:
    print("pytorch3d     : import failed; set HYVLA_REBUILD_PYTORCH3D=1 if RoboTwin needs it:", repr(exc), flush=True)
else:
    show("pytorch3d", pytorch3d)
PY

cat <<EOF

Done. Next quick test:

conda activate ${ENV_NAME}
cd ${HY_VLA_DIR}
HYVLA_REQUIRE_SITE_CUROBO=1 \\
TASKS_OVERRIDE=adjust_bottle \\
TEST_NUM=1 \\
ROBOTWIN_DIR=${ROBOTWIN_DIR} \\
CKPT_PATH=${HY_VLA_DIR}/ckpts/Hy-VLA-RoboTwin \\
CUDA_VISIBLE_DEVICES=0 \\
bash scripts/eval_robotwin_test.sh

EOF
