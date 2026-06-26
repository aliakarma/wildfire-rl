#!/usr/bin/env python
"""Strip outputs and execution counts from every notebook in-place.

A dependency-free fallback for nbstripout (which is wired into pre-commit). The original
notebooks embedded multi-MB base64 plot outputs; stripping them shrinks the repo and
makes notebook diffs reviewable.

    python scripts/strip_notebooks.py [notebooks_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def strip_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
    nb.get("metadata", {}).pop("widgets", None)
    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks")
    nbs = sorted(root.rglob("*.ipynb"))
    if not nbs:
        print(f"No notebooks under {root}")
        return 0
    n = sum(strip_notebook(p) for p in nbs)
    print(f"Stripped {n}/{len(nbs)} notebooks under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
