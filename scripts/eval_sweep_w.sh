#!/usr/bin/env bash
# =============================================================================
# Sweep the language-prior guidance weight w over a RoboTwin eval.
#
# For each w in W_GRID it sets HYVLA_GUIDANCE_W (read by build_policy in
# robotwin_eval/policy_wrapper.py) and runs the 6-task regression, writing each
# w's per-task logs under eval_logs/w_<w>/. The core hypothesis test: does OOD
# success rate rise as w drops below 1.0?
#
# Usage (single GPU):
#   ROBOTWIN_DIR=/path/to/RoboTwin \
#   CKPT_PATH=$(pwd)/ckpts/Hy-VLA-RoboTwin \
#   TASK_CONFIG=demo_randomized TEST_NUM=20 \
#   bash scripts/eval_sweep_w.sh
#
# Env (with defaults):
#   W_GRID         "1.0 0.75 0.5 0.25"
#   TASK_CONFIG    demo_randomized   (OOD; use demo_clean for the ID control)
#   plus everything eval_robotwin_test.sh reads: ROBOTWIN_DIR, CKPT_PATH,
#   TEST_NUM, CUDA_VISIBLE_DEVICES, INSTRUCTION_TYPE, SEED.
# =============================================================================

set -euo pipefail

HY_VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W_GRID=${W_GRID:-"1.0 0.75 0.5 0.25"}
export TASK_CONFIG=${TASK_CONFIG:-demo_randomized}

echo "========================================================"
echo "Hy-VLA guidance-w sweep"
echo "w grid       : ${W_GRID}"
echo "task_config  : ${TASK_CONFIG}"
echo "========================================================"

for w in ${W_GRID}; do
    export HYVLA_GUIDANCE_W="${w}"
    export LOG_DIR="${HY_VLA_DIR}/eval_logs/${TASK_CONFIG}/w_${w}"
    echo
    echo "########## guidance_w=${w}  ->  ${LOG_DIR} ##########"
    bash "${HY_VLA_DIR}/scripts/eval_robotwin_test.sh" || echo "[w=${w}] eval returned nonzero; see ${LOG_DIR}"
done

echo
echo "[sweep done] per-w logs under ${HY_VLA_DIR}/eval_logs/${TASK_CONFIG}/"
echo "Collect: grep -r 'Success rate' ${HY_VLA_DIR}/eval_logs/${TASK_CONFIG}/"
