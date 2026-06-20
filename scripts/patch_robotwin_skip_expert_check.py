#!/usr/bin/env python3
"""Patch RoboTwin eval_policy.py to skip expert play_once seed filtering.

RoboTwin's eval loop normally runs the task's expert ``play_once`` before the
policy rollout to find a successful seed and synthesize an instruction. That
path uses cuRobo planning and can fail before a VLA policy is ever called. This
debug patch bypasses the expert check and falls back to a simple instruction
derived from the task name when the description generator has no expert info.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_eval_policy(path: Path) -> None:
    text = path.read_text()
    backup = path.with_suffix(path.suffix + ".hyvla_skip_expert.bak")
    if not backup.exists():
        backup.write_text(text)

    patched = text
    patched = patched.replace("expert_check = True", "expert_check = False  # Hy-VLA debug: skip expert play_once")

    if "episode_info = {'info': {}}" not in patched:
        patched = patched.replace(
            "task_total_reward = 0\n",
            "task_total_reward = 0\n"
            "    episode_info = {'info': {}}  # Hy-VLA debug: fallback when expert_check is skipped\n",
        )

    old = (
        '    episode_info_list = [episode_info["info"]]\n'
        '    results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)\n'
        "    instruction = np.random.choice(results[0][instruction_type])\n"
    )
    new = (
        "    try:\n"
        '        episode_info_list = [episode_info["info"]]\n'
        '        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)\n'
        "        instruction = np.random.choice(results[0][instruction_type])\n"
        "    except Exception:\n"
        '        instruction = args["task_name"].replace("_", " ")\n'
        '        print(f"[Hy-VLA debug] fallback instruction: {instruction}", flush=True)\n'
    )
    if old in patched and "[Hy-VLA debug] fallback instruction" not in patched:
        patched = patched.replace(old, new)

    if patched == text:
        print(f"[Hy-VLA debug] expert-check skip patch already applied: {path}")
    else:
        path.write_text(patched)
        print(f"[Hy-VLA debug] Patched RoboTwin expert-check skip: {path}")
    print(f"[Hy-VLA debug] Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-dir", required=True)
    args = parser.parse_args()
    eval_policy = Path(args.robotwin_dir) / "script" / "eval_policy.py"
    if not eval_policy.exists():
        raise SystemExit(f"eval_policy.py not found: {eval_policy}")
    patch_eval_policy(eval_policy)


if __name__ == "__main__":
    main()
