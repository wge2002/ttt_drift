#!/usr/bin/env python3
"""Print the cuRobo package layout that RoboTwin's planner depends on."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path


def spec_text(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "<missing>"
    origin = spec.origin or "<namespace>"
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return f"{origin}; package_dirs={locations}"
    return origin


def main() -> None:
    print("python", sys.executable)

    import curobo

    root = Path(curobo.__file__).resolve().parent
    print("curobo", getattr(curobo, "__file__", "<unknown>"))
    print("curobo root", root)

    names = [
        "curobo.types",
        "curobo.types.math",
        "curobo.types.base",
        "curobo.types.robot",
        "curobo.types.state",
        "curobo.types.camera",
        "curobo.wrap.reacher.motion_gen",
        "curobo.cuda_robot_model.cuda_robot_model",
        "curobo.geom.sdf.world",
    ]
    for name in names:
        print(f"{name}: {spec_text(name)}")

    try:
        types_mod = importlib.import_module("curobo.types")
    except Exception as exc:
        print("import curobo.types failed:", repr(exc))
    else:
        print("curobo.types object", types_mod)
        print("curobo.types file", getattr(types_mod, "__file__", "<no file>"))
        print("curobo.types path", getattr(types_mod, "__path__", "<no path>"))

    print("top-level curobo modules:")
    for mod in pkgutil.iter_modules([str(root)]):
        if mod.name.startswith(("types", "wrap", "geom", "cuda", "util")):
            print(" ", mod.name, "pkg" if mod.ispkg else "module")


if __name__ == "__main__":
    main()
