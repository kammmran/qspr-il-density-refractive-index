import pandas as pd

from qspr_il import cli
from qspr_il.registry import get as get_spec


def _write_input_csv(tmp_path, rows):
    path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_main_end_to_end_mixture_model(tmp_path, monkeypatch, fake_ensemble_dir):
    # No --mole_fraction_col passed: relies on spec.default_mole_fraction_col
    # ("Mole_fraction_IL", matching the bundled datasets/external_test_set.csv convention).
    input_csv = _write_input_csv(
        tmp_path, {"SMILES": ["CCO.[Cl-]"], "Mole_fraction_IL": [0.4]})
    output_csv = tmp_path / "out.csv"

    argv = [
        "--model",
        "5",
        "--input_csv",
        str(input_csv),
        "--model_dir",
        str(fake_ensemble_dir),
        "--output_csv",
        str(output_csv),
    ]
    rc = cli.main(argv)
    assert rc == 0
    assert output_csv.exists()
    result = pd.read_csv(output_csv)
    assert "prediction_mean" in result.columns


def test_main_end_to_end_pure_model_ignores_mole_fraction(tmp_path, fake_ensemble_dir):
    input_csv = _write_input_csv(tmp_path, {"SMILES": ["C[N+](C)(C)C.[Cl-]"]})
    output_csv = tmp_path / "out.csv"

    argv = [
        "--model",
        "8",
        "--input_csv",
        str(input_csv),
        "--model_dir",
        str(fake_ensemble_dir),
        "--output_csv",
        str(output_csv),
    ]
    rc = cli.main(argv)
    assert rc == 0
    result = pd.read_csv(output_csv)
    assert "Mole_fraction" not in result.columns
    assert "prediction_mean" in result.columns


def test_missing_model_triggers_interactive_prompt(monkeypatch, tmp_path, fake_ensemble_dir):
    input_csv = _write_input_csv(tmp_path, {"SMILES": ["C[N+](C)(C)C.[Cl-]"]})
    output_csv = tmp_path / "out.csv"

    answers = iter(
        [
            "1",  # choose action: run a prediction model
            "8",  # choose model at the menu
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))

    argv = [
        "--input_csv",
        str(input_csv),
        "--model_dir",
        str(fake_ensemble_dir),
        "--output_csv",
        str(output_csv),
    ]
    rc = cli.main(argv)
    assert rc == 0
    assert output_csv.exists()


def test_main_reports_clear_error_for_mismatched_mole_fraction_column(tmp_path, fake_ensemble_dir, capsys):
    # Regression test: a wrong --mole_fraction_col used to silently produce an all-NaN
    # output CSV with no error message. It should now fail fast with a clear message.
    input_csv = _write_input_csv(
        tmp_path, {"SMILES": ["CCO.[Cl-]"], "Mole_fraction_IL": [0.4]})
    output_csv = tmp_path / "out.csv"

    argv = [
        "--model",
        "5",
        "--input_csv",
        str(input_csv),
        "--mole_fraction_col",
        "Mole_fraction",  # does not exist in the input CSV -- it's named Mole_fraction_IL
        "--model_dir",
        str(fake_ensemble_dir),
        "--output_csv",
        str(output_csv),
    ]
    rc = cli.main(argv)
    assert rc == 1
    assert not output_csv.exists()
    assert "Mole_fraction" in capsys.readouterr().out


def test_resolve_args_pure_model_never_touches_mole_fraction():
    spec = get_spec("8")
    args = cli.build_parser().parse_args(
        ["--input_csv", "in.csv", "--model_dir", "d", "--output_csv", "out.csv"])
    resolved = cli.resolve_args(args, spec)
    assert resolved.mole_fraction_col is None


def _fake_curated_df(property_name="Density", n=2):
    return pd.DataFrame(
        {
            "IL_SMILES": ["C[N+](C)(C)C.[Cl-]"] * n,
            "Temperature (K)": [298.15] * n,
            "Property": [property_name] * n,
            "Property_value": [1000.0] * n,
        }
    )


def test_action_data_fetches_and_saves_curated_csv(monkeypatch, tmp_path, capsys):
    output_csv = tmp_path / "custom" / "density_pure.csv"
    answers = iter(
        [
            "2",  # action: fetch data
            "density",  # property
            "",  # solvent: blank -> pure
            "none",  # temperature range: no limit
            "none",  # pressure range: no limit
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "10",  # max datasets
            str(output_csv),  # output path
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    monkeypatch.setattr("qspr_il.data.cleaning.fetch_curated_dataset",
                        lambda *a, **k: _fake_curated_df())

    rc = cli.main([])
    assert rc == 0
    assert output_csv.exists()
    saved = pd.read_csv(output_csv)
    assert len(saved) == 2
    assert set(saved["Property"]) == {"Density"}
    assert "Saved 2 curated rows" in capsys.readouterr().out


def test_action_data_binary_system_prompts_for_one_solvent(monkeypatch, tmp_path):
    output_csv = tmp_path / "custom" / "density_ethanol.csv"
    answers = iter(
        [
            "2",  # action: fetch data
            "density",  # property
            "2",  # system type: binary mixture
            "ethanol",  # solvent 1 name
            "none",  # temperature range
            "none",  # pressure range
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "10",  # max datasets
            str(output_csv),  # output path
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    calls = []

    def fake_fetch(*a, **k):
        calls.append((a, k))
        return _fake_curated_df()

    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", fake_fetch)

    rc = cli.main([])
    assert rc == 0
    assert output_csv.exists()
    (property_query, solvent_name, solvent_name_2), _kwargs = calls[0]
    assert solvent_name == "ethanol"
    assert solvent_name_2 is None


def test_action_data_ternary_system_prompts_for_two_solvents(monkeypatch, tmp_path):
    output_csv = tmp_path / "custom" / "density_ethanol_water.csv"
    answers = iter(
        [
            "2",  # action: fetch data
            "density",  # property
            "3",  # system type: ternary mixture
            "ethanol",  # solvent 1 name
            "water",  # solvent 2 name
            "none",  # temperature range
            "none",  # pressure range
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "10",  # max datasets
            str(output_csv),  # output path
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    calls = []

    def fake_fetch(*a, **k):
        calls.append((a, k))
        return _fake_curated_df()

    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", fake_fetch)

    rc = cli.main([])
    assert rc == 0
    assert output_csv.exists()
    (property_query, solvent_name, solvent_name_2), _kwargs = calls[0]
    assert solvent_name == "ethanol"
    assert solvent_name_2 == "water"


def test_action_data_all_systems_option_fetches_without_solvent_prompts(monkeypatch, tmp_path):
    output_csv = tmp_path / "custom" / "density_all.csv"
    answers = iter(
        [
            "2",  # action: fetch data
            "density",  # property
            "0",  # system type: all compounds and mixtures
            "none",  # temperature range
            "none",  # pressure range
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "10",  # max datasets
            str(output_csv),  # output path
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    calls = []

    def fake_fetch(*a, **k):
        calls.append((a, k))
        return _fake_curated_df()

    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", fake_fetch)

    rc = cli.main([])
    assert rc == 0
    assert output_csv.exists()
    (property_query, solvent_name, solvent_name_2), kwargs = calls[0]
    assert solvent_name is None
    assert solvent_name_2 is None
    assert kwargs["all_systems"] is True


def test_action_data_reports_error_for_unmatched_property(monkeypatch, capsys):
    answers = iter(["2", "not-a-real-property", "",
                   "none", "none", "", "", "", "5", "out.csv"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))

    def raise_value_error(*a, **k):
        raise ValueError("No property matching 'not-a-real-property' found.")

    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", raise_value_error)

    rc = cli.main([])
    assert rc == 1
    assert "No property matching" in capsys.readouterr().out


def test_action_both_chains_into_matching_prediction_model(monkeypatch, tmp_path, fake_ensemble_dir):
    output_data_csv = tmp_path / "density_pure.csv"
    output_pred_csv = tmp_path / "pred.csv"
    answers = iter(
        [
            "3",  # action: both
            "density",  # property
            "",  # solvent: blank -> pure
            "none",  # temperature range
            "none",  # pressure range
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "5",  # max datasets
            str(output_data_csv),  # data output path
            "Y",  # run the matching model now
            str(output_pred_csv),  # predictions output path
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    monkeypatch.setattr("qspr_il.data.cleaning.fetch_curated_dataset",
                        lambda *a, **k: _fake_curated_df(n=1))
    monkeypatch.setattr(
        "qspr_il.cli.find_spec",
        lambda property_name, solvent_label: get_spec("8"),  # density, pure IL
    )
    from qspr_il.models.engine import load_models_and_metadata as real_load_models_and_metadata

    monkeypatch.setattr(
        "qspr_il.cli.load_models_and_metadata", lambda model_dir: real_load_models_and_metadata(
            fake_ensemble_dir)
    )

    rc = cli.main([])
    assert rc == 0
    assert output_pred_csv.exists()
    result = pd.read_csv(output_pred_csv)
    assert "prediction_mean" in result.columns


def test_action_both_skips_prediction_when_no_matching_model(monkeypatch, tmp_path):
    output_data_csv = tmp_path / "viscosity_pure.csv"
    answers = iter(
        [
            "3",  # action: both
            "viscosity",  # property with no trained model
            "",  # solvent: blank -> pure
            "none",
            "none",
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "5",
            str(output_data_csv),
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", lambda *a, **k: _fake_curated_df(
            property_name="Viscosity", n=1)
    )

    rc = cli.main([])
    assert rc == 0


_TRAINING_SMILES_POOL = [
    "CCCCn1cc[n+](C)c1.F[B-](F)(F)F",
    "CCN1C=C[N+](=C1)C.F[B-](F)(F)F",
    "CCCCN1C=C[N+](=C1)C.[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "CCCCCCN1C=C[N+](=C1)C.F[P-](F)(F)(F)(F)F",
]


def _fake_curated_df_for_training():
    return pd.DataFrame(
        {
            "IL_SMILES": _TRAINING_SMILES_POOL * 2,
            "Temperature (K)": [298.15, 303.15] * 4,
            "Property": ["Viscosity"] * 8,
            "Property_value": [1.0 + 0.1 * i for i in range(8)],
        }
    )


def test_action_both_auto_trains_and_predicts_when_no_matching_model(monkeypatch, tmp_path):
    output_data_csv = tmp_path / "viscosity_pure.csv"
    output_model_dir = tmp_path / "viscosity_pure_model"
    output_pred_csv = tmp_path / "viscosity_pred.csv"
    answers = iter(
        [
            "3",  # action: both
            "viscosity",  # property with no trained model
            "",  # solvent: blank -> pure
            "none",
            "none",
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "5",
            str(output_data_csv),
            "y",  # train a new model now
            str(output_model_dir),  # save trained model to
            "2",  # number of ensemble members
            "Y",  # run this newly trained model now
            str(output_pred_csv),  # save predictions to
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", lambda *a, **k: _fake_curated_df_for_training()
    )

    rc = cli.main([])
    assert rc == 0
    assert (output_model_dir / "model_1.joblib").exists()
    assert (output_model_dir / "model_2.joblib").exists()
    assert output_pred_csv.exists()
    result = pd.read_csv(output_pred_csv)
    assert "prediction_mean" in result.columns
    assert not result["prediction_mean"].isna().any()


def test_action_both_can_decline_training_after_no_matching_model(monkeypatch, tmp_path):
    output_data_csv = tmp_path / "viscosity_pure.csv"
    answers = iter(
        [
            "3",  # action: both
            "viscosity",
            "",
            "none",
            "none",
            "",  # search filter: year
            "",  # search filter: author
            "",  # search filter: keyword
            "5",
            str(output_data_csv),
            "n",  # decline training
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    monkeypatch.setattr(
        "qspr_il.data.cleaning.fetch_curated_dataset", lambda *a, **k: _fake_curated_df_for_training()
    )

    rc = cli.main([])
    assert rc == 0


def _forbid_input(monkeypatch):
    def _no_input(*_args, **_kwargs):
        raise AssertionError(
            "the CLI prompted interactively when every needed option was given")

    monkeypatch.setattr("builtins.input", _no_input)


def test_model_dir_without_model_runs_the_custom_ensemble(tmp_path, monkeypatch, fake_ensemble_dir):
    # Regression: the command _offer_auto_train prints after training a custom model
    # ("--model_dir ... --input_csv ... --smiles_col ...", no --model) dropped into the
    # interactive model menu instead of running, then died on EOF in a non-interactive shell.
    _forbid_input(monkeypatch)
    input_csv = _write_input_csv(
        tmp_path, {"IL_SMILES": ["C[N+](C)(C)C.[Cl-]"], "Temperature (K)": [298.15]})
    output_csv = tmp_path / "out.csv"

    rc = cli.main(
        [
            "--model_dir", str(fake_ensemble_dir),
            "--input_csv", str(input_csv),
            "--smiles_col", "IL_SMILES",
            "--temp_col", "Temperature (K)",
            "--output_csv", str(output_csv),
        ]
    )
    assert rc == 0
    result = pd.read_csv(output_csv)
    assert not result["prediction_mean"].isna().any()


def test_model_dir_without_model_is_a_mixture_model_when_a_mole_fraction_column_is_given(
    tmp_path, monkeypatch, fake_ensemble_dir
):
    _forbid_input(monkeypatch)
    input_csv = _write_input_csv(
        tmp_path,
        {"IL_SMILES": ["C[N+](C)(C)C.[Cl-]"], "Temperature (K)": [
            298.15], "Mole_fraction_IL": [0.4]},
    )
    output_csv = tmp_path / "out.csv"

    rc = cli.main(
        [
            "--model_dir", str(fake_ensemble_dir),
            "--input_csv", str(input_csv),
            "--smiles_col", "IL_SMILES",
            "--temp_col", "Temperature (K)",
            "--mole_fraction_col", "Mole_fraction_IL",
            "--output_csv", str(output_csv),
        ]
    )
    assert rc == 0
    result = pd.read_csv(output_csv)
    # The mixture-composition feature was carried through under its canonical name.
    assert "Mole_fraction" in result.columns
    assert not result["prediction_mean"].isna().any()
