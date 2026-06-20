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
#   HYVLA_PATCH_ROBOTWIN_TRACEBACK 1 to patch RoboTwin's swallowed exceptions
#   HYVLA_PATCH_CUROBO_NO_GRAPH 1 to disable cuRobo CUDA graph warmup
#   HYVLA_PATCH_CUROBO_SKIP_WARMUP 1 to skip cuRobo warmup calls entirely
#   HYVLA_PATCH_SKIP_EXPERT_CHECK 1 to skip RoboTwin expert play_once precheck
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

if [ "${HYVLA_PATCH_ROBOTWIN_TRACEBACK:-0}" = "1" ]; then
    ROBOTWIN_EVAL_POLICY="${ROBOTWIN_DIR}/script/eval_policy.py"
    python - "${ROBOTWIN_EVAL_POLICY}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "traceback.print_exc()" in text:
    print(f"[Hy-VLA debug] RoboTwin traceback patch already present: {path}", flush=True)
    raise SystemExit(0)

backup = path.with_suffix(path.suffix + ".hyvla_traceback.bak")
if not backup.exists():
    backup.write_text(text)

lines = text.splitlines(keepends=True)
out = []
inserted = False
imported = False
for line in lines:
    out.append(line)
    stripped = line.strip()
    if stripped == "import traceback" or stripped.startswith("import traceback,"):
        imported = True
    if stripped.startswith("import ") and not imported:
        out.append("import traceback\n")
        imported = True
    if "error occurs" in line and "print" in line:
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}traceback.print_exc()\n")
        inserted = True

if not inserted:
    raise SystemExit(f"[Hy-VLA debug] Did not find 'error occurs' print in {path}")

path.write_text("".join(out))
print(f"[Hy-VLA debug] Patched RoboTwin traceback printing: {path}", flush=True)
print(f"[Hy-VLA debug] Backup: {backup}", flush=True)
PY
fi

if [ "${HYVLA_PATCH_SKIP_EXPERT_CHECK:-0}" = "1" ]; then
    python "${HY_VLA_DIR}/scripts/patch_robotwin_skip_expert_check.py" \
        --robotwin-dir "${ROBOTWIN_DIR}"
fi

if [ "${HYVLA_PATCH_CUROBO_NO_GRAPH:-0}" = "1" ] || [ "${HYVLA_PATCH_CUROBO_SKIP_WARMUP:-0}" = "1" ]; then
    patch_args=(--robotwin-dir "${ROBOTWIN_DIR}")
    if [ "${HYVLA_PATCH_CUROBO_SKIP_WARMUP:-0}" = "1" ]; then
        patch_args+=(--skip-warmup)
    fi
    python "${HY_VLA_DIR}/scripts/patch_robotwin_curobo_no_graph.py" "${patch_args[@]}"
fi

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
echo "Hy-VLA git    : $(git -C "${HY_VLA_DIR}" rev-parse --short HEAD 2>/dev/null || echo '<unknown>')"
echo "Policy link   : $(readlink -f "${ROBOTWIN_DIR}/policy/hy_vla" 2>/dev/null || echo '<unresolved>')"
if grep -q "Exception inside RoboTwin eval hook" "${ROBOTWIN_DIR}/policy/hy_vla/deploy_policy.py" 2>/dev/null; then
    echo "Hook traceback : enabled"
else
    echo "Hook traceback : MISSING"
fi
if grep -q "traceback.print_exc()" "${ROBOTWIN_DIR}/script/eval_policy.py" 2>/dev/null; then
    echo "RoboTwin trace : enabled"
else
    echo "RoboTwin trace : disabled"
fi
if grep -q "use_cuda_graph=False" "${ROBOTWIN_DIR}/envs/robot/planner.py" 2>/dev/null; then
    echo "cuRobo graph   : disabled"
else
    echo "cuRobo graph   : default"
fi
if grep -q "skip cuRobo warmup" "${ROBOTWIN_DIR}/envs/robot/planner.py" 2>/dev/null; then
    echo "cuRobo warmup  : skipped"
else
    echo "cuRobo warmup  : default"
fi
if grep -q "skip expert play_once" "${ROBOTWIN_DIR}/script/eval_policy.py" 2>/dev/null; then
    echo "Expert precheck: skipped"
else
    echo "Expert precheck: default"
fi
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
