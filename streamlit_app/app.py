"""Streamlit Community Cloud entry point -- thin shim over ``qspr_il.app``.

Streamlit Cloud clones this repository to the deployment machine and runs this
file directly, so the ``qspr_il`` package lives one directory up rather than
being pip-installed. Add the repo root to ``sys.path`` before importing it.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qspr_il.app import main

if __name__ == "__main__":
    main()
