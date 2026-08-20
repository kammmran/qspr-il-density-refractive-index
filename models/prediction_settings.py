"""Interactive configuration shared by the model application scripts."""


def configure_prediction_args(args, default_model_dir, default_output_csv="predictions.csv"):
    """Ask for missing prediction settings and return the updated arguments."""
    if args.input_csv is not None:
        return args

    print("\nPrediction settings (press Enter to use the default in brackets).")
    args.input_csv = _prompt("Input CSV path", "")
    while not args.input_csv:
        print("Input CSV path is required.")
        args.input_csv = _prompt("Input CSV path", "")

    args.smiles_col = _prompt("SMILES column", args.smiles_col or "SMILES")
    args.mole_fraction_col = _prompt(
        "Mole fraction column", args.mole_fraction_col or "Mole_fraction"
    )
    args.temp_col = _prompt(
        "Temperature column (leave empty for 298.15 K)", args.temp_col or ""
    )
    args.model_dir = _prompt(
        "Model directory", args.model_dir or default_model_dir)
    args.output_csv = _prompt(
        "Output CSV path", args.output_csv or default_output_csv)
    print()
    return args


def _prompt(label, default):
    if default:
        value = input(f"{label} [{default}]: ").strip()
        return value or default
    return input(f"{label}: ").strip()
