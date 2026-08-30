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
    input_csv = _write_input_csv(tmp_path, {"SMILES": ["CCO.[Cl-]"], "Mole_fraction_IL": [0.4]})
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
    input_csv = _write_input_csv(tmp_path, {"SMILES": ["CCO.[Cl-]"], "Mole_fraction_IL": [0.4]})
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
    args = cli.build_parser().parse_args(["--input_csv", "in.csv", "--model_dir", "d", "--output_csv", "out.csv"])
    resolved = cli.resolve_args(args, spec)
    assert resolved.mole_fraction_col is None
