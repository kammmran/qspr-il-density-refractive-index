# QSPR: Ionic-Liquid Density & Refractive Index -- Streamlit app export

This folder is the deployment entry point for the Streamlit Community Cloud
app at https://qspr-il-density-refractive-index.streamlit.app/

Predict density (kg/m3) and refractive index (Na D-line) of ionic-liquid
mixtures (in water, ethanol, or isopropanol) and pure ionic liquids, using
trained XGBoost ensemble models.

Choose a property and system in the sidebar, then either upload a CSV of
ionic-liquid SMILES or enter a single SMILES string directly.

## Contents

- `app.py` -- thin shim: adds the repo root to `sys.path` and calls
  `qspr_il.app.main()`.
- `requirements.txt` -- third-party runtime dependencies. `qspr_il` is imported
  from the repository checkout (with its bundled trained model artifacts), not
  pip-installed.

## Deployment

On Streamlit Community Cloud, point the app at this repository and set the main
module path to `streamlit_app/app.py`. Streamlit Cloud clones the whole repo, so
the `qspr_il` package and its packaged models are available without a separate
install step.

Source code, model training details, and the underlying data pipeline are
documented in the project repository:
https://github.com/kammmran/qspr-il-density-refractive-index
