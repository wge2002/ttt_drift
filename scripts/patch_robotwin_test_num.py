#!/usr/bin/env python3
"""Patch RoboTwin eval_policy.py so --test_num overrides the hard-coded 100."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = "    test_num = 100\n"
NEW = "    test_num = int(usr_args.get(\"test_num\", 100))  # Hy-VLA: honor CLI override\n"


def patch_eval_policy(path: Path) -> None:
    text = path.read_text()
    backup = path.with_suffix(path.suffix + ".hyvla_test_num.bak")
    if not backup.exists():
        backup.write_text(text)

    if NEW in text:
        print(f"[Hy-VLA debug] test_num override patch already applied: {path}")
        print(f"[Hy-VLA debug] Backup: {backup}")
        return

    if OLD not in text:
        raise SystemExit(f"Could not find hard-coded test_num line in {path}")

    path.write_text(text.replace(OLD, NEW, 1))
    print(f"[Hy-VLA debug] Patched RoboTwin test_num override: {path}")
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
