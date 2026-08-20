"""Interactive launcher for the ionic-liquid QSPR prediction models."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = "datasets/external_test_set.csv"
DEFAULT_SMILES_COLUMN = "IL_SMILES"
DEFAULT_MOLE_FRACTION_COLUMN = "Mole_fraction_IL"

MODELS = {
    "1": ("Refractive index", "ethanol", "models/ri_ethanol/apply_ri_ethanol.py", "models/ri_ethanol/RI_ethanol_ensemble_model", False),
    "2": ("Refractive index", "isopropanol", "models/ri_isopropanol/apply_ri_isopropanol.py", "models/ri_isopropanol/RI_isopropanol_ensemble_model", False),
    "3": ("Refractive index", "water", "models/ri_water/apply_ri_water.py", "models/ri_water/RI_water_ensemble_model", False),
    "4": ("Refractive index", "pure ionic liquid", "models/ri_pure/apply_ri_pure.py", "models/ri_pure/RI_pure_ensemble_model", True),
    "5": ("Density", "ethanol", "models/density_ethanol/apply_density_ethanol.py", "models/density_ethanol/density_ethanol_ensemble_model", False),
    "6": ("Density", "isopropanol", "models/density_isopropanol/apply_density_isopropanol.py", "models/density_isopropanol/density_isopropanol_ensemble_model", False),
    "7": ("Density", "water", "models/density_water/apply_density_water.py", "models/density_water/density_water_ensemble_model", False),
    "8": ("Density", "pure ionic liquid", "models/density_pure/apply_density_pure.py", "models/density_pure/density_pure_ensemble_model", True),
}


def prompt(label, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def choose_model():
    print("\nChoose a prediction model:")
    for key, (property_name, solvent, *_rest) in MODELS.items():
        print(f"  {key}. {property_name} in {solvent}")

    while True:
        choice = prompt("Model", "1")
        if choice in MODELS:
            return MODELS[choice]
        print("Please select a number from 1 to 8.")


def build_command(model, input_csv, smiles_col, temp_col, model_dir, output_csv, mole_fraction_col):
    _property_name, _solvent, script, _default_model_dir, is_pure = model
    command = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--input_csv",
        input_csv,
        "--smiles_col",
        smiles_col,
        "--temp_col",
        temp_col,
        "--model_dir",
        model_dir,
        "--output_csv",
        output_csv,
    ]
    if not is_pure:
        command.extend(["--mole_fraction_col", mole_fraction_col])
    return command


def main():
    property_name, solvent, script, default_model_dir, is_pure = choose_model()
    model_name = Path(script).parent.name
    default_output_csv = f"results/{model_name}_prediction.csv"
    print(f"\nSelected: {property_name} in {solvent}")
    print("Press Enter to keep the default shown in brackets.\n")

    input_csv = prompt("Input CSV", DEFAULT_INPUT)
    while not input_csv:
        print("Input CSV is required.")
        input_csv = prompt("Input CSV", DEFAULT_INPUT)

    smiles_col = prompt("SMILES column", DEFAULT_SMILES_COLUMN)
    temp_col = prompt("Temperature column", "")
    mole_fraction_col = ""
    if not is_pure:
        mole_fraction_col = prompt("Mole fraction column", DEFAULT_MOLE_FRACTION_COLUMN)
    model_dir = prompt("Model directory", default_model_dir)
    output_csv = prompt("Output CSV", default_output_csv)
    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(
        (property_name, solvent, script, default_model_dir, is_pure),
        input_csv,
        smiles_col,
        temp_col,
        model_dir,
        str(output_path),
        mole_fraction_col,
    )
    print("\nStarting prediction...\n")
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive launcher for the QSPR prediction models.")
    parser.parse_args()
    raise SystemExit(main())