#!/usr/bin/env python3
"""Launch the DNDSR Cauchy Monitor Hub from any checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
PY_DIR = ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from DNDSR.Cauchy.monitor.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
