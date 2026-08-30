"""Command-line entry point for running QSPR predictions.

Replaces the old ``qspr.py`` menu + ``subprocess`` dispatch and the 8 duplicated
``apply_*.py`` argparse blocks. Everything runs in-process against
:mod:`qspr_il.models.engine`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qspr_il.models.engine import load_models_and_metadata, run_prediction
from qspr_il.registry import ModelSpec, iter_specs
from qspr_il.registry import get as get_spec


def _prompt(label: str, default: str = "") -> str:
    if default:
        value = input(f"{label} [{default}]: ").strip()
        return value or default
    return input(f"{label}: ").strip()


def _interactive_choose_model() -> ModelSpec:
    print("\nChoose a prediction model:")
    for spec in iter_specs():
        print(f"  {spec.key}. {spec.label}")
    while True:
        choice = _prompt("Model", "1")
        try:
            return get_spec(choice)
        except KeyError:
            print("Please select a number from 1 to 8.")


def resolve_args(args: argparse.Namespace, spec: ModelSpec) -> argparse.Namespace:
    """Fill in any unset CLI arguments interactively, using ``spec`` for defaults.

    Only asks for a mole-fraction column when the model is not a pure-IL model.
    """
    if args.input_csv is None:
        print("\nPrediction settings (press Enter to use the default in brackets).")
        args.input_csv = _prompt("Input CSV path", "datasets/external_test_set.csv")
        while not args.input_csv:
            print("Input CSV path is required.")
            args.input_csv = _prompt("Input CSV path", "")
        args.smiles_col = _prompt("SMILES column", args.smiles_col or spec.default_smiles_col)
        if not spec.is_pure:
            args.mole_fraction_col = _prompt(
                "Mole fraction column", args.mole_fraction_col or spec.default_mole_fraction_col
            )
        args.temp_col = _prompt("Temperature column (leave empty for 298.15 K)", args.temp_col or "")
        args.model_dir = _prompt("Model directory", args.model_dir or str(spec.model_dir))
        default_output = f"results/{Path(spec.model_dir).parent.name}_prediction.csv"
        args.output_csv = _prompt("Output CSV path", args.output_csv or default_output)
        print()
    else:
        if not args.smiles_col:
            args.smiles_col = spec.default_smiles_col
        if not spec.is_pure and not args.mole_fraction_col:
            args.mole_fraction_col = spec.default_mole_fraction_col
        if not args.model_dir:
            args.model_dir = str(spec.model_dir)
        if not args.output_csv:
            args.output_csv = f"results/{Path(spec.model_dir).parent.name}_prediction.csv"

    if spec.is_pure:
        args.mole_fraction_col = None

    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict density or refractive index of ionic liquids.")
    parser.add_argument("--model", choices=sorted(r.key for r in iter_specs()), help="Model to run (1-8).")
    parser.add_argument("--input_csv", help="Path to the input CSV file (if omitted, you will be prompted).")
    parser.add_argument("--smiles_col", help="Name of the SMILES column in the input CSV.")
    parser.add_argument("--mole_fraction_col", help="Name of the mole fraction column (mixture models only).")
    parser.add_argument("--temp_col", help="Name of the temperature column.")
    parser.add_argument("--model_dir", help="Directory containing the ensemble of models and metadata.")
    parser.add_argument("--output_csv", help="Path to save the output CSV with predictions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = get_spec(args.model) if args.model else _interactive_choose_model()
    args = resolve_args(args, spec)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        return 1

    data = pd.read_csv(input_path)
    try:
        ensemble = load_models_and_metadata(args.model_dir)
    except FileNotFoundError as e:
        print(f"Error loading models or metadata: {e}")
        return 1

    try:
        result = run_prediction(
            data,
            spec,
            ensemble=ensemble,
            smiles_col=args.smiles_col,
            temp_col=args.temp_col,
            mole_fraction_col=args.mole_fraction_col,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
