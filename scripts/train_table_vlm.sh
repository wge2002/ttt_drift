#!/usr/bin/env bash
# =============================================================================
# Hy-VLA UMI pretraining: single-table recipe.
#
# Same as ``train_umi_vlm.sh`` but trains on ONE Lance table only.
# Useful for fast iteration / debugging or when you only need a subset.
#
# Usage (run on EACH node; one MUST set INDEX=0 and act as chief):
#
#   # Node 0 (chief):
#   export CHIEF_IP=<chief-ip>  INDEX=0  TABLE_NAME=table_001
#   bash scripts/train_umi_vlm_single_table.sh
#
#   # Node N (worker):
#   export CHIEF_IP=<chief-ip>  INDEX=N  TABLE_NAME=table_001
#   bash scripts/train_umi_vlm_single_table.sh
#
# Env overrides (with defaults shown):
#   EXP_ID            hy_vlm_bootstrap_umi_lance_single
#   EXP_ROOT          /path/to/experiments
#   VLM_PATH          tencent/HY-Embodied-0.5 (override with local path)
#   LANCE_SOURCE      tencent/Hy-Embodied-0.5-VLA-Data (override with local lance_source)
#   NORM_PATH         <EXP_ROOT>/<EXP_ID>/norm_stats.pkl
#   TABLE_NAME        table_001
#   NUM_MACHINES      8
#   NPROC_PER_NODE    8
#   MAIN_PORT         6688
#   HY_VLA_DIR        parent of this script
#   CHIEF_IP          (required)
#   INDEX             (required)  rank in [0, NUM_MACHINES)
# =============================================================================

set -euo pipefail

# --------- 1. CLI / env ---------
HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

EXP_ID=${EXP_ID:-"hy_vlm_bootstrap_umi_lance_single"}
EXP_ROOT=${EXP_ROOT:-"/path/to/experiments"}
VLM_PATH=${VLM_PATH:-"tencent/HY-Embodied-0.5"}
LANCE_SOURCE=${LANCE_SOURCE:-tencent/Hy-Embodied-0.5-VLA-Data}
NORM_PATH=${NORM_PATH:-"${EXP_ROOT}/${EXP_ID}/norm_stats.pkl"}
TABLE_NAME=${TABLE_NAME:-"table_001"}

NUM_MACHINES=${NUM_MACHINES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAIN_PORT=${MAIN_PORT:-6688}

: "${CHIEF_IP:?CHIEF_IP must be set (rank-0 node IP)}"
: "${INDEX:?INDEX must be set (this node rank in [0, NUM_MACHINES))}"

NUM_PROCESSES=$(( NUM_MACHINES * NPROC_PER_NODE ))
CKPT_SAVE_DIR="${EXP_ROOT}/${EXP_ID}"

# --------- 2. Banner ---------
echo "========================================================"
echo "Hy-VLA UMI Lance pretraining (VLM bootstrap, SINGLE TABLE)"
echo "EXP_ID         : ${EXP_ID}"
echo "ckpt_save_dir  : ${CKPT_SAVE_DIR}"
echo "pretrain (vlm) : ${VLM_PATH}"
echo "lance_source   : ${LANCE_SOURCE}"
echo "lance_tables   : ${TABLE_NAME}"
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
    dataset=umi_lance \
    dataset.lance_source="${LANCE_SOURCE}" \
    dataset.lance_tables="${TABLE_NAME}" \
    dataset.mean_std_path="${NORM_PATH}" \
    training.batch_size=16
