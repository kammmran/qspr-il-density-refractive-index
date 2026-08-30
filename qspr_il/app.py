"""Streamlit GUI for the QSPR density/refractive-index prediction models.

Run locally with::

    streamlit run qspr_il/app.py

For Hugging Face Spaces deployment, see ``huggingface_space/`` at the repo
root and :doc:`/streamlit_app` in the Sphinx docs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from qspr_il.models.engine import load_models_and_metadata, run_prediction
from qspr_il.registry import ModelSpec, iter_specs

_BANNER_IMAGE = Path(__file__).resolve().parent / "assets" / "il_github.png"


@st.cache_resource(show_spinner="Loading trained ensemble...")
def _load_ensemble(model_dir: str):
    return load_models_and_metadata(model_dir)


def _select_spec() -> ModelSpec:
    specs = iter_specs()
    properties = sorted({s.property_name for s in specs})
    property_name = st.sidebar.selectbox("Property", properties)
    solvents = [s.solvent for s in specs if s.property_name == property_name]
    solvent = st.sidebar.selectbox("System", solvents)
    return next(s for s in specs if s.property_name == property_name and s.solvent == solvent)


def _render_report(result: pd.DataFrame, spec: ModelSpec) -> None:
    """Summary analysis of a batch prediction run: KPIs, distribution, and uncertainty."""
    st.subheader("Prediction report")

    valid = result["prediction_mean"].notna()
    invalid_smiles = (result.get("Changes") == "Invalid SMILES") if "Changes" in result.columns else pd.Series(
        False, index=result.index
    )
    std_series = result.loc[valid, "prediction_std"]
    high_uncertainty_threshold = std_series.quantile(0.75) if len(std_series) else float("nan")

    cols = st.columns(4)
    cols[0].metric("Rows predicted", f"{int(valid.sum())} / {len(result)}")
    cols[1].metric(f"Mean {spec.target_column}", f"{result.loc[valid, 'prediction_mean'].mean():.4f}" if valid.any() else "n/a")
    cols[2].metric("Mean ensemble std", f"{std_series.mean():.4f}" if len(std_series) else "n/a")
    cols[3].metric("Unparseable SMILES", int(invalid_smiles.sum()))

    if valid.sum() == 0:
        st.warning("No valid predictions to analyze -- check the SMILES/column settings above.")
        return

    st.markdown(f"**Distribution of predicted {spec.target_column}**")
    counts, bin_edges = np.histogram(result.loc[valid, "prediction_mean"], bins=min(20, max(3, valid.sum())))
    hist_df = pd.DataFrame(
        {"count": counts},
        index=[f"{bin_edges[i]:.3g}-{bin_edges[i + 1]:.3g}" for i in range(len(bin_edges) - 1)],
    )
    st.bar_chart(hist_df)

    st.markdown("**Prediction vs. ensemble uncertainty**")
    st.scatter_chart(result.loc[valid], x="prediction_mean", y="prediction_std")

    top_uncertain = result.loc[valid].sort_values("prediction_std", ascending=False).head(5)
    with st.expander(f"5 highest-uncertainty predictions (std > {high_uncertainty_threshold:.4g})"):
        st.dataframe(top_uncertain)

    if invalid_smiles.any():
        with st.expander(f"{int(invalid_smiles.sum())} row(s) with unparseable SMILES (excluded above)"):
            st.dataframe(result.loc[invalid_smiles])


def _run_csv_mode(spec: ModelSpec) -> None:
    uploaded = st.file_uploader("Input CSV", type="csv")
    smiles_col = st.text_input("SMILES column", value=spec.default_smiles_col)
    mole_fraction_col = None
    if not spec.is_pure:
        mole_fraction_col = st.text_input("Mole fraction column", value=spec.default_mole_fraction_col)
    temp_col = st.text_input("Temperature column (leave empty for 298.15 K default)", value="")

    if uploaded is None:
        st.info("Upload a CSV to run predictions.")
        return

    if st.button("Run prediction", type="primary"):
        try:
            data = pd.read_csv(uploaded)
            ensemble = _load_ensemble(str(spec.model_dir))
            result = run_prediction(
                data,
                spec,
                ensemble=ensemble,
                smiles_col=smiles_col,
                temp_col=temp_col or None,
                mole_fraction_col=mole_fraction_col,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        st.session_state["csv_result"] = (result, spec.key)

    cached = st.session_state.get("csv_result")
    if cached is None or cached[1] != spec.key:
        return
    result, _ = cached

    st.dataframe(result)
    st.download_button(
        "Download predictions CSV",
        result.to_csv(index=False),
        file_name=f"{spec.property_name.lower().replace(' ', '_')}_{spec.solvent.replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )
    _render_report(result, spec)


def _run_single_entry_mode(spec: ModelSpec) -> None:
    smiles = st.text_input("Ionic liquid SMILES", value="")
    temperature = st.number_input("Temperature (K)", value=298.15)
    mole_fraction = None
    if not spec.is_pure:
        mole_fraction = st.number_input("IL mole fraction", min_value=0.0, max_value=1.0, value=0.5)

    if st.button("Predict", type="primary"):
        if not smiles.strip():
            st.error("Enter a SMILES string.")
            return
        row = {"SMILES": [smiles], "Temperature": [temperature]}
        if mole_fraction is not None:
            row["Mole_fraction"] = [mole_fraction]
        data = pd.DataFrame(row)

        try:
            ensemble = _load_ensemble(str(spec.model_dir))
            result = run_prediction(
                data,
                spec,
                ensemble=ensemble,
                smiles_col="SMILES",
                temp_col="Temperature",
                mole_fraction_col="Mole_fraction" if mole_fraction is not None else None,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        mean = result.loc[0, "prediction_mean"]
        std = result.loc[0, "prediction_std"]
        st.metric(f"Predicted {spec.target_column}", f"{mean} ± {std}")
        if result.loc[0, "Changes"] != "No changes":
            st.caption(f"SMILES standardization: {result.loc[0, 'Changes']}")


def main() -> None:
    st.set_page_config(page_title="QSPR: IL Density & Refractive Index", page_icon="\U0001F9EA")
    if _BANNER_IMAGE.exists():
        st.image(str(_BANNER_IMAGE), width="stretch")
    st.title("QSPR: Ionic-Liquid Density & Refractive Index")
    st.write(
        "Predict density (kg/m3) and refractive index (Na D-line) of ionic-liquid "
        "mixtures and pure ionic liquids using trained XGBoost ensemble models."
    )

    spec = _select_spec()
    st.subheader(spec.label)
    st.caption(spec.description)

    mode = st.radio("Input mode", ["Upload CSV", "Single IL entry"], horizontal=True)
    if mode == "Upload CSV":
        _run_csv_mode(spec)
    else:
        _run_single_entry_mode(spec)


if __name__ == "__main__":
    main()
