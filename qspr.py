"""Backward-compatible entry point: ``python qspr.py``.

The real implementation lives in :mod:`ilqspr.cli`; this file exists so the
command documented in the README keeps working.
"""

from ilqspr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
