#!/usr/bin/env python3
"""Patch RoboTwin planner imports for newer nvidia-curobo package layouts.

RoboTwin's planner imports ``Pose`` and ``JointState`` from old submodules such
as ``curobo.types.math``. Newer nvidia-curobo releases expose these symbols from
``curobo.types`` instead, where ``curobo.types`` is a module rather than a
package. This patch keeps the old path first, then falls back to the new public
export.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_POSE = "from curobo.types.math import Pose as CuroboPose"
NEW_POSE = """try:
    from curobo.types.math import Pose as CuroboPose
except (ImportError, ModuleNotFoundError):
    from curobo.types import Pose as CuroboPose"""

OLD_JOINT_STATE = "from curobo.types.robot import JointState"
NEW_JOINT_STATE = """try:
    from curobo.types.robot import JointState
except (ImportError, ModuleNotFoundError):
    from curobo.types import JointState"""


def patch_planner(planner_path: Path) -> None:
    text = planner_path.read_text()
    backup = planner_path.with_suffix(planner_path.suffix + ".hyvla_curobo_v2_api.bak")
    if not backup.exists():
        backup.write_text(text)

    patched = text
    replacements = 0
    for old, new in [(OLD_POSE, NEW_POSE), (OLD_JOINT_STATE, NEW_JOINT_STATE)]:
        if old in patched:
            patched = patched.replace(old, new)
            replacements += 1

    if patched == text:
        print(f"[Hy-VLA debug] cuRobo v2 import patch already applied or not needed: {planner_path}")
    else:
        planner_path.write_text(patched)
        print(f"[Hy-VLA debug] Patched cuRobo v2 imports in planner: {planner_path}")
        print(f"[Hy-VLA debug] Import replacements: {replacements}")
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
