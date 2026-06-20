#!/usr/bin/env python3
"""Patch RoboTwin's cuRobo planner to disable CUDA graph warmup.

This is a local debugging/workaround patch for H20/Hopper CUDA illegal
instruction failures inside cuRobo's planner warmup. It edits
``<ROBOTWIN_DIR>/envs/robot/planner.py`` in place and saves a backup next to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    in_string: str | None = None
    escaped = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in ('"', "'"):
            in_string = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError("Could not find matching ')' in planner.py")


def _patch_motion_gen_config(text: str) -> tuple[str, int]:
    needle = "MotionGenConfig.load_from_robot_config("
    pos = 0
    count = 0
    pieces: list[str] = []
    while True:
        start = text.find(needle, pos)
        if start < 0:
            pieces.append(text[pos:])
            break
        open_idx = text.find("(", start)
        close_idx = _find_matching_paren(text, open_idx)
        block = text[start : close_idx + 1]
        if "use_cuda_graph" not in block:
            last_newline = block.rfind("\n")
            if last_newline < 0:
                raise ValueError("Unexpected one-line MotionGenConfig call")
            closing_line = block[last_newline + 1 :]
            indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
            block = block[:last_newline] + f"\n{indent}use_cuda_graph=False,\n" + closing_line
            count += 1
        pieces.append(text[pos:start])
        pieces.append(block)
        pos = close_idx + 1
    return "".join(pieces), count


def patch_planner(planner_path: Path) -> None:
    text = planner_path.read_text()
    backup = planner_path.with_suffix(planner_path.suffix + ".hyvla_no_graph.bak")
    if not backup.exists():
        backup.write_text(text)

    patched, config_count = _patch_motion_gen_config(text)
    warmup_count = 0

    replacements = {
        "self.motion_gen.warmup()": (
            "self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)"
        ),
        "self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)": (
            "self.motion_gen_batch.warmup("
            "batch=CONFIGS.ROTATE_NUM, enable_graph=False, warmup_js_trajopt=False)"
        ),
    }
    for old, new in replacements.items():
        if old in patched:
            patched = patched.replace(old, new)
            warmup_count += 1

    if patched == text:
        print(f"[Hy-VLA debug] cuRobo no-graph patch already applied: {planner_path}")
    else:
        planner_path.write_text(patched)
        print(f"[Hy-VLA debug] Patched cuRobo no-graph planner: {planner_path}")
        print(f"[Hy-VLA debug] MotionGenConfig calls patched: {config_count}")
        print(f"[Hy-VLA debug] warmup calls patched: {warmup_count}")
    print(f"[Hy-VLA debug] Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-dir", required=True)
    args = parser.parse_args()
    planner_path = Path(args.robotwin_dir) / "envs" / "robot" / "planner.py"
    if not planner_path.exists():
        raise SystemExit(f"planner.py not found: {planner_path}")
    patch_planner(planner_path)


if __name__ == "__main__":
    main()
