import numpy as np
import pandas as pd
import pytest

from qspr_il.models.engine import (
    calculate_descriptors,
    load_models_and_metadata,
    predict,
    prepare_input,
    process_il_smiles_list,
    reorder_charged_species,
    run_prediction,
    standardize_molecule,
)
from qspr_il.registry import get as get_spec


def test_standardize_molecule_valid_smiles():
    standardized, summary = standardize_molecule("CCO")
    assert standardized
    assert summary == "No changes" or "Reionized" in summary or "normalized" in summary.lower()


def test_standardize_molecule_invalid_smiles():
    standardized, summary = standardize_molecule("not a smiles")
    assert standardized == "not a smiles"
    assert summary == "Invalid SMILES"


def test_reorder_charged_species_cation_first():
    df = pd.DataFrame({"Standardized_IL_SMILES": ["[Cl-].[Na+]"]})
    result = reorder_charged_species(df)
    assert result.loc[0, "Standardized_IL_SMILES"] == "[Na+].[Cl-]"


def test_load_models_and_metadata_happy_path(fake_ensemble_dir):
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    assert len(ensemble.models) == 5
    assert len(ensemble.metadata) == 5
    assert ensemble.metadata[0]["descriptors"] == ["nAtom", "nHeavyAtom"]


@pytest.mark.parametrize("missing_index", [1, 3])
def test_load_models_and_metadata_gap_in_numbering_raises(fake_ensemble_dir, missing_index):
    # A gap (a later-numbered model still present after a missing one) is always corruption --
    # unlike deleting the highest-numbered file, which just means a smaller valid ensemble.
    (fake_ensemble_dir / f"model_{missing_index}.joblib").unlink()
    with pytest.raises(FileNotFoundError):
        load_models_and_metadata(fake_ensemble_dir)


def test_load_models_and_metadata_no_models_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models_and_metadata(tmp_path)


def test_load_models_and_metadata_supports_fewer_than_five(fake_ensemble_dir):
    # A freshly trained ensemble (qspr_il.models.training) may have fewer than 5 members if
    # there weren't enough unique compounds for a full 5-fold split -- a contiguous run
    # starting at 1, however short, is a valid ensemble.
    (fake_ensemble_dir / "model_5.joblib").unlink()
    (fake_ensemble_dir / "metadata_5.json").unlink()
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    assert len(ensemble.models) == 4
    assert len(ensemble.metadata) == 4


def test_calculate_descriptors_shape_matches_required():
    required = ["nAtom", "nHeavyAtom"]
    vector, names = calculate_descriptors("CCO", required)
    assert vector is not None
    assert len(vector) == len(required)


def test_calculate_descriptors_invalid_smiles_returns_none():
    vector, names = calculate_descriptors("not a smiles", ["nAtom"])
    assert vector is None
    assert names is None


def test_process_il_smiles_list_invalid_row_is_nan():
    required = ["nAtom", "nHeavyAtom"]
    result = process_il_smiles_list(["CCO", "not a smiles"], required)
    assert result.shape == (2, len(required))
    assert not np.isnan(result[0]).any()
    assert np.isnan(result[1]).all()


def test_prepare_input_mixture_defaults_missing_temperature():
    df = pd.DataFrame({"SMILES": ["CCO"], "Mole_fraction": [0.5]})
    prepared = prepare_input(df, smiles_col="SMILES",
                             temp_col=None, mole_fraction_col="Mole_fraction")
    assert prepared.loc[0, "Temperature"] == 298.15
    assert "Standardized_IL_SMILES" in prepared.columns
    assert prepared.loc[0, "Changes"] == "No changes"


def test_prepare_input_pure_skips_mole_fraction():
    df = pd.DataFrame({"SMILES": ["CCO"]})
    prepared = prepare_input(df, smiles_col="SMILES",
                             temp_col=None, mole_fraction_col=None)
    assert "Mole_fraction" not in prepared.columns


def test_prepare_input_missing_mole_fraction_column_raises_clear_error():
    # Regression test: a mismatched mole_fraction_col used to be silently ignored by
    # DataFrame.rename (a no-op on an unknown key), causing every ensemble member to fail
    # downstream with the real cause never surfaced to the caller.
    df = pd.DataFrame({"SMILES": ["CCO"], "Mole_fraction_IL": [0.5]})
    with pytest.raises(ValueError, match="Mole_fraction"):
        prepare_input(df, smiles_col="SMILES", temp_col=None,
                      mole_fraction_col="Mole_fraction")


def test_prepare_input_missing_smiles_column_raises_clear_error():
    df = pd.DataFrame({"IL_SMILES": ["CCO"]})
    with pytest.raises(ValueError, match="SMILES"):
        prepare_input(df, smiles_col="SMILES",
                      temp_col=None, mole_fraction_col=None)


def test_prepare_input_flags_invalid_smiles_in_changes_column():
    df = pd.DataFrame({"SMILES": ["not a smiles"]})
    prepared = prepare_input(df, smiles_col="SMILES",
                             temp_col=None, mole_fraction_col=None)
    assert prepared.loc[0, "Changes"] == "Invalid SMILES"


def test_run_prediction_output_includes_changes_column(fake_ensemble_dir, sample_pure_prediction_df):
    from qspr_il.registry import get as get_spec

    spec = get_spec("8")
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    result = run_prediction(sample_pure_prediction_df,
                            spec, ensemble=ensemble, smiles_col="SMILES")
    assert "Changes" in result.columns


def test_predict_per_model_failure_degrades_to_nan(fake_ensemble_dir, sample_prediction_df):
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    # Break one model so its predict() raises.
    ensemble.models[2].predict = lambda X: (
        _ for _ in ()).throw(RuntimeError("boom"))

    prepared = prepare_input(
        sample_prediction_df, smiles_col="SMILES", temp_col="Temperature", mole_fraction_col="Mole_fraction"
    )
    result = predict(prepared, ensemble, mole_fraction_col="Mole_fraction")

    assert "prediction_mean" in result.columns
    assert "prediction_std" in result.columns
    assert not result["prediction_mean"].isna().any()


def test_run_prediction_mixture_spec(fake_ensemble_dir, sample_prediction_df):
    spec = get_spec("5")  # Density in ethanol (mixture)
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    result = run_prediction(
        sample_prediction_df, spec, ensemble=ensemble, smiles_col="SMILES", mole_fraction_col="Mole_fraction"
    )
    assert "prediction_mean" in result.columns
    assert len(result) == len(sample_prediction_df)


def test_run_prediction_pure_spec_excludes_mole_fraction(fake_ensemble_dir, sample_pure_prediction_df):
    spec = get_spec("8")  # Density, pure IL
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    result = run_prediction(sample_pure_prediction_df,
                            spec, ensemble=ensemble, smiles_col="SMILES")
    assert "prediction_mean" in result.columns
    assert len(result) == len(sample_pure_prediction_df)


def test_run_prediction_reports_progress(fake_ensemble_dir, sample_pure_prediction_df):
    spec = get_spec("8")
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    messages = []
    run_prediction(
        sample_pure_prediction_df, spec, ensemble=ensemble, smiles_col="SMILES", progress_callback=messages.append
    )
    assert any("Standardizing" in m for m in messages)
    assert any("member 1/5" in m for m in messages)
    assert any("member 5/5" in m for m in messages)
    assert any("Combining ensemble predictions" in m for m in messages)


def test_run_prediction_reports_ensemble_loading_when_not_cached(fake_ensemble_dir, sample_pure_prediction_df):
    spec = get_spec("8")
    messages = []
    run_prediction(
        sample_pure_prediction_df,
        spec,
        # avoid touching the real model_dir
        ensemble=load_models_and_metadata(fake_ensemble_dir),
        smiles_col="SMILES",
        progress_callback=messages.append,
    )
    # ensemble already provided -> no "Loading trained ensemble" message expected
    assert not any("Loading trained ensemble" in m for m in messages)


def test_predict_reports_progress_on_model_failure(fake_ensemble_dir, sample_prediction_df):
    ensemble = load_models_and_metadata(fake_ensemble_dir)
    ensemble.models[2].predict = lambda X: (
        _ for _ in ()).throw(RuntimeError("boom"))
    prepared = prepare_input(
        sample_prediction_df, smiles_col="SMILES", temp_col="Temperature", mole_fraction_col="Mole_fraction"
    )
    messages = []
    predict(prepared, ensemble, mole_fraction_col="Mole_fraction",
            progress_callback=messages.append)
    assert any("Model 3/5: prediction failed" in m for m in messages)


def test_reorder_charged_species_keeps_neutral_components():
    # Regression: components with no formal charge used to be dropped entirely, so a neutral
    # input collapsed to an empty string and a hydrate lost its water -- both of which then
    # produced all-NaN descriptors and a meaningless prediction.
    df = pd.DataFrame(
        {"Standardized_IL_SMILES": ["CCO", "O.[Na+].[Cl-]", "CCO.C[N+](C)(C)C.[Cl-]"]})
    result = reorder_charged_species(df)["Standardized_IL_SMILES"].tolist()
    assert result == ["CCO", "[Na+].[Cl-].O", "C[N+](C)(C)C.[Cl-].CCO"]


def test_reorder_charged_species_passes_through_unparseable_smiles():
    # standardize_molecule returns an invalid SMILES unchanged; reordering must not mangle it
    # into an empty string on the way through.
    df = pd.DataFrame({"Standardized_IL_SMILES": ["not a smiles"]})
    assert reorder_charged_species(
        df).loc[0, "Standardized_IL_SMILES"] == "not a smiles"


def test_reorder_charged_species_handles_multiply_charged_cation():
    df = pd.DataFrame({"Standardized_IL_SMILES": ["[Cl-].[Cl-].[Fe+2]"]})
    assert reorder_charged_species(
        df).loc[0, "Standardized_IL_SMILES"] == "[Fe+2].[Cl-].[Cl-]"


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _two_member_ensemble():
    from qspr_il.models.engine import LoadedEnsemble

    return LoadedEnsemble(
        models=[_ConstantModel(1.0), _ConstantModel(3.0)],
        metadata=[{"descriptors": ["nAtom"]}, {"descriptors": ["nAtom"]}],
    )


def test_predict_does_not_align_on_the_input_index():
    # Regression: the per-member predictions carry a fresh 0..n-1 index, so assigning them as
    # Series let pandas align them against the caller's index -- any input whose index isn't a
    # plain range (e.g. a filtered DataFrame) came back entirely NaN.
    data = pd.DataFrame(
        {
            "Standardized_IL_SMILES": ["CCO", "CCO", "CCO"],
            "Temperature": [298.15, 300.0, 310.0],
            "Mole_fraction": [0.5, 0.5, 0.5],
        },
        index=[10, 11, 12],
    )
    result = predict(data, _two_member_ensemble())
    assert result["prediction_mean"].tolist() == [2.0, 2.0, 2.0]
    assert not result["prediction_std"].isna().any()


def test_predict_coerces_non_numeric_conditions_instead_of_failing_every_member():
    # One unparseable cell used to make the whole column object dtype, which the estimator
    # rejects -- failing every ensemble member and leaving the entire run NaN. Only the bad
    # row should be affected now.
    data = pd.DataFrame(
        {
            "Standardized_IL_SMILES": ["CCO", "CCO"],
            "Temperature": ["298.15", "n/a"],
            "Mole_fraction": [0.5, 0.5],
        }
    )
    result = predict(data, _two_member_ensemble())
    assert result["prediction_mean"].tolist() == [2.0, 2.0]
