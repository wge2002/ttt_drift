#!/usr/bin/env bash
# =============================================================================
# Hy-VLA RoboTwin eval: regress on the first-6 tasks (10 rollouts each)
# on a single GPU.
#
# Usage:
#   bash scripts/eval_robotwin_test.sh
#
# Env overrides (with defaults):
#   TEST_NUM       10
#   CUDA_VISIBLE_DEVICES  0
#   CKPT_PATH      /path/to/Hy-VLA-RoboTwin
#   ROBOTWIN_DIR   /path/to/RoboTwin
#   HY_VLA_DIR     parent of this script
#   LOG_DIR        <HY_VLA_DIR>/eval_logs
#   TASKS_OVERRIDE optional whitespace-separated task list, e.g. "adjust_bottle"
#   HYVLA_SKIP_PREFLIGHT 1 to skip Python import/dependency preflight
# =============================================================================

set -euo pipefail

# --------- 1. defaults ---------
TEST_NUM=${TEST_NUM:-10}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-/path/to/RoboTwin}
CKPT_PATH=${CKPT_PATH:-/path/to/Hy-VLA-RoboTwin}
LOG_DIR=${LOG_DIR:-"${HY_VLA_DIR}/eval_logs"}

# --------- 2. Fixed first-6 task list (regression subset) ---------
if [ -n "${TASKS_OVERRIDE:-}" ]; then
    read -r -a TASKS <<<"${TASKS_OVERRIDE}"
else
    TASKS=(
        adjust_bottle
        beat_block_hammer
        blocks_ranking_rgb
        blocks_ranking_size
        click_alarmclock
        click_bell
    )
fi

# --------- 3. RoboTwin eval_policy.py invariants for Hy-VLA ---------
CKPT_SETTING=Hy-VLA-RoboTwin
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
SEED=${SEED:-10000}

# --------- 4. Symlink robotwin_eval/ -> RoboTwin/policy/hy_vla (idempotent) ---
ln -sfn "${HY_VLA_DIR}/robotwin_eval" "${ROBOTWIN_DIR}/policy/hy_vla"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${HY_VLA_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
    export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi

# --------- 5. Banner ---------
echo "========================================================"
echo "Hy-VLA RoboTwin quick regression (single GPU)"
echo "Tasks (${#TASKS[@]})    : ${TASKS[*]}"
echo "Rollouts/task : ${TEST_NUM}"
echo "GPU           : ${CUDA_VISIBLE_DEVICES}"
echo "Ckpt path     : ${CKPT_PATH}"
echo "RoboTwin dir  : ${ROBOTWIN_DIR}"
echo "Log dir       : ${LOG_DIR}"
echo "Vulkan ICD    : ${VK_ICD_FILENAMES:-<unset>}"
echo "Driver caps   : ${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
echo "========================================================"

# --------- 6. Hy-VLA import/dependency preflight ---------
if [ "${HYVLA_SKIP_PREFLIGHT:-0}" != "1" ]; then
    python - <<'PY'
import importlib.util
import sys

print("Python        :", sys.executable, flush=True)

try:
    import transformers
except Exception as exc:
    print("transformers import failed:", repr(exc), flush=True)
    raise SystemExit(1)

print(
    "transformers  :",
    getattr(transformers, "__version__", "<unknown>"),
    getattr(transformers, "__file__", "<unknown>"),
    flush=True,
)

missing = [
    name
    for name in ["transformers.modeling_layers", "timm", "flash_attn"]
    if importlib.util.find_spec(name) is None
]
if missing:
    print("Missing Hy-VLA runtime module(s):", ", ".join(missing), flush=True)
    print(
        'Install these inside the dedicated RoboTwinHy conda env, not RLinf .venv: '
        'pip install -U "transformers>=4.57,<4.58" '
        'safetensors "huggingface-hub>=0.23" timm==1.0.21',
        flush=True,
    )
    print(
        "If flash_attn is missing too, install the wheel matching this Python/Torch/CUDA env.",
        flush=True,
    )
    raise SystemExit(1)

try:
    import hy_vla  # noqa: F401
except Exception as exc:
    print("hy_vla import failed:", repr(exc), flush=True)
    raise

print("Hy-VLA import : OK", flush=True)
PY
fi

# --------- 7. Sequential evaluation loop ---------
n_done=0
n_failed=0
for task in "${TASKS[@]}"; do
    log="${LOG_DIR}/${task}.log"
    echo "[${task}] starting ...  (log: ${log})"
    rc=0
    (
        cd "${ROBOTWIN_DIR}"
        PYTHONWARNINGS=ignore::UserWarning \
        python -u script/eval_policy.py \
            --config policy/hy_vla/deploy_policy.yml \
            --overrides \
                --task_name "${task}" \
                --task_config "${TASK_CONFIG}" \
                --ckpt_setting "${CKPT_SETTING}" \
                --instruction_type "${INSTRUCTION_TYPE}" \
                --seed "${SEED}" \
                --test_num "${TEST_NUM}" \
                --policy_name policy.hy_vla \
                --ckpt_path "${CKPT_PATH}" \
            2>&1 \
            | sed -u 's/\r/\n/g' \
            | sed -u '/^\(\x1b\[[0-9;]*m\)*step: /d' \
            >"${log}"
    ) || rc=$?
    if [ $rc -eq 0 ]; then
        rate=$(grep -a 'Success rate' "${log}" | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' || true)
        echo "[${task}] done.  ${rate}"
        n_done=$((n_done + 1))
    else
        echo "[${task}] FAILED (rc=${rc}); see ${log}"
        n_failed=$((n_failed + 1))
    fi
done

echo "--------------------------------------------------------"
echo "Finished: ${n_done}/${#TASKS[@]} tasks succeeded"
[ $n_failed -gt 0 ] && echo "WARNING: ${n_failed} task(s) failed — check logs in ${LOG_DIR}"
[ $n_failed -gt 0 ] && exit 1
exit 0
