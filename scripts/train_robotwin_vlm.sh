#!/usr/bin/env bash
# =============================================================================
# Hy-VLA RoboTwin training: 2-node x 8-GPU recipe (VLM-bootstrap variant).
#
# This launcher loads ONLY the upstream Hy-Embodied VLM backbone into
# the inner ``dual_tower.vlm`` slot; the action expert is randomly
# initialized. Use this when training a brand-new Hy-VLA from scratch
# on top of the public ``tencent/HY-Embodied-0.5`` release. To instead
# fine-tune a released Hy-VLA checkpoint, use ``train_robotwin_umi.sh``.
#
# Usage (run on EACH node; one MUST set INDEX=0 and act as chief):
#
#   # Node 0 (chief):
#   export CHIEF_IP=<chief-ip>  INDEX=0
#   bash scripts/train_robotwin_vlm.sh
#
#   # Node 1 (worker):
#   export CHIEF_IP=<chief-ip>  INDEX=1
#   bash scripts/train_robotwin_vlm.sh
#
# Env overrides (with defaults shown):
#   EXP_ID            hy_vlm_bootstrap_robotwin
#   EXP_ROOT          /path/to/experiments
#   VLM_PATH          tencent/HY-Embodied-0.5 (override with local path)
#   HDF5_DIR          /path/to/robotwin/hdf5
#   NORM_PATH         <EXP_ROOT>/<EXP_ID>/norm_stats.pkl
#   NUM_MACHINES      2
#   NPROC_PER_NODE    8
#   MAIN_PORT         6688
#   HY_VLA_DIR        parent of this script
#   CHIEF_IP          (required)
#   INDEX             (required)  rank in [0, NUM_MACHINES)
# =============================================================================

set -euo pipefail

# --------- 1. CLI / env ---------
HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

EXP_ID=${EXP_ID:-"hy_vlm_bootstrap_robotwin"}
EXP_ROOT=${EXP_ROOT:-"/path/to/experiments"}
VLM_PATH=${VLM_PATH:-"tencent/HY-Embodied-0.5"}
HDF5_DIR=${HDF5_DIR:-"/path/to/robotwin/hdf5"}
NORM_PATH=${NORM_PATH:-"${EXP_ROOT}/${EXP_ID}/norm_stats.pkl"}

NUM_MACHINES=${NUM_MACHINES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAIN_PORT=${MAIN_PORT:-6688}

: "${CHIEF_IP:?CHIEF_IP must be set (rank-0 node IP)}"
: "${INDEX:?INDEX must be set (this node rank in [0, NUM_MACHINES))}"

NUM_PROCESSES=$(( NUM_MACHINES * NPROC_PER_NODE ))
CKPT_SAVE_DIR="${EXP_ROOT}/${EXP_ID}"

# --------- 2. Banner ---------
echo "========================================================"
echo "Hy-VLA RoboTwin training (VLM bootstrap)"
echo "EXP_ID         : ${EXP_ID}"
echo "ckpt_save_dir  : ${CKPT_SAVE_DIR}"
echo "pretrain (vlm) : ${VLM_PATH}"
echo "hdf5_dir       : ${HDF5_DIR}"
echo "norm_stats.pkl : ${NORM_PATH}"
echo "topology       : ${NUM_MACHINES} nodes x ${NPROC_PER_NODE} gpus = ${NUM_PROCESSES} procs"
echo "this node      : INDEX=${INDEX}  CHIEF_IP=${CHIEF_IP}  PORT=${MAIN_PORT}"
echo "========================================================"

# --------- 3. Launch ---------
cd "${HY_VLA_DIR}"

accelerate launch \
    --multi_gpu \
    --num_machines "${NUM_MACHINES}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${CHIEF_IP}" \
    --main_process_port "${MAIN_PORT}" \
    --machine_rank "${INDEX}" \
    hy_vla/train.py \
    exp_id="${EXP_ID}" \
    exp_name="${EXP_ID}" \
    ckpt_save_dir="${CKPT_SAVE_DIR}" \
    model.pretrain_source=vlm \
    model.vlm_model_path="${VLM_PATH}" \
    dataset.hdf5_dir="${HDF5_DIR}" \
    dataset.mean_std_path="${NORM_PATH}"
