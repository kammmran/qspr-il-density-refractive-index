"""Consolidated QSPR prediction engine.

Replaces the 8 near-duplicate ``apply_*.py`` scripts that used to live under
``models/<name>/``. All of them shared the same standardization, descriptor
calculation, and ensemble-prediction logic; this module keeps that logic in
one place, parametrized by a :class:`~qspr_il.registry.ModelSpec`.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors as mordred_descriptors
from mordred.error import Missing
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

if TYPE_CHECKING:
    from qspr_il.registry import ModelSpec

DEFAULT_TEMPERATURE_K = 298.15


def standardize_molecule(smi: str) -> tuple[str, str]:
    """Reionize, normalize functional groups, and strip stereochemistry from a SMILES string.

    Returns ``(standardized_smiles, change_summary)``. Invalid or unparseable SMILES are
    returned unchanged along with a description of what went wrong.
    """
    changes = []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, "Invalid SMILES"

    try:
        parts = smi.split(".")
        reionized_parts = []
        for part in parts:
            part_mol = Chem.MolFromSmiles(part)
            if part_mol:
                reionized = rdMolStandardize.Reionize(part_mol)
                if Chem.MolToSmiles(reionized) != Chem.MolToSmiles(part_mol):
                    changes.append("Reionized")
                reionized_parts.append(Chem.MolToSmiles(reionized))
            else:
                reionized_parts.append(part)
        reionized_smi = ".".join(reionized_parts)
        if reionized_smi != smi:
            mol = Chem.MolFromSmiles(reionized_smi)
        else:
            mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        return smi, f"Reionization Error: {e}"

    try:
        normalizer = rdMolStandardize.Normalizer()
        normalized = normalizer.normalize(mol)
        if Chem.MolToSmiles(normalized) != Chem.MolToSmiles(mol):
            changes.append("Functional groups normalized")
        mol = normalized
    except Exception as e:
        return smi, f"Normalization Error: {e}"

    try:
        smi_before_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
        smi_after_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smi_before_stereo != smi_after_stereo:
            changes.append("Stereo removed")
    except Exception as e:
        return smi, f"Stereo Cleanup Error: {e}"

    standardized_smi = Chem.MolToSmiles(mol)
    change_summary = ", ".join(changes) if changes else "No changes"
    return standardized_smi, change_summary


def reorder_charged_species(df: pd.DataFrame, smiles_col: str = "Standardized_IL_SMILES") -> pd.DataFrame:
    """Reorder each multi-component SMILES so cations come before anions."""

    def reorder_smiles(smi):
        parts = smi.split(".")
        cations = []
        anions = []
        for part in parts:
            if re.search(r"\+[0-9]*", part):
                cations.append(part)
            elif re.search(r"\-[0-9]*", part):
                anions.append(part)
        return ".".join(cations + anions)

    df[smiles_col] = df[smiles_col].apply(reorder_smiles)
    return df


@dataclass
class LoadedEnsemble:
    """The 5 trained models + their metadata for one :class:`ModelSpec`."""

    models: list
    metadata: list[dict]


def load_models_and_metadata(model_dir: str | Path) -> LoadedEnsemble:
    """Load the 5-model ensemble (``model_1..5.joblib`` + ``metadata_1..5.json``) from a directory."""
    model_dir = Path(model_dir)
    models = []
    model_metadata = []
    for i in range(1, 6):
        model_path = model_dir / f"model_{i}.joblib"
        metadata_path = model_dir / f"metadata_{i}.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model or metadata file for model {i} not found in {model_dir}.")
        models.append(joblib.load(model_path))
        with open(metadata_path, "r") as f:
            model_metadata.append(json.load(f))
    return LoadedEnsemble(models=models, metadata=model_metadata)


def calculate_descriptors(smiles: str, required_descriptors: list[str]) -> tuple[np.ndarray | None, list[str] | None]:
    """Compute the mean Mordred descriptor vector across all components of a multi-part SMILES."""
    smiles_list = smiles.split(".")
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in required_descriptors]
    descriptor_vectors = []
    descriptor_names = None
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc = calc(mol)
            if descriptor_names is None:
                descriptor_names = [str(d) for d in desc.keys()]
            desc_vector = [np.nan if isinstance(value, Missing) else value for _, value in desc.items()]
            descriptor_vectors.append(desc_vector)
        else:
            print(f"Invalid SMILES: {smi}")
    if descriptor_vectors:
        descriptor_vectors = np.array(descriptor_vectors, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            mean_descriptor_vector = np.nanmean(descriptor_vectors, axis=0)
        return mean_descriptor_vector, descriptor_names
    print(f"No valid molecules found for SMILES: {smiles}")
    return None, None


def process_il_smiles_list(smiles_list: list[str], required_descriptors: list[str]) -> np.ndarray:
    """Compute descriptor vectors for a whole column of (possibly multi-component) SMILES."""
    descriptor_vectors = []
    for smiles in smiles_list:
        mean_descriptor_vector, _ = calculate_descriptors(smiles, required_descriptors)
        if mean_descriptor_vector is not None:
            descriptor_vectors.append(mean_descriptor_vector)
        else:
            descriptor_vectors.append([np.nan] * len(required_descriptors))
    return np.array(descriptor_vectors)


def prepare_input(
    data: pd.DataFrame,
    smiles_col: str,
    temp_col: str | None,
    mole_fraction_col: str | None,
    default_temp: float = DEFAULT_TEMPERATURE_K,
) -> pd.DataFrame:
    """Standardize SMILES and normalize the Temperature/Mole_fraction columns in place.

    ``mole_fraction_col`` should be ``None`` for pure-IL model specs (no mixture composition).
    """
    data = data.copy()

    if smiles_col not in data.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in input data. Available columns: {list(data.columns)}")

    standardized_smiles = []
    changes = []
    for smi in data[smiles_col]:
        standardized_smi, change_summary = standardize_molecule(smi)
        standardized_smiles.append(standardized_smi)
        changes.append(change_summary)
    data["Standardized_IL_SMILES"] = standardized_smiles
    data["Changes"] = changes
    data = reorder_charged_species(data, smiles_col="Standardized_IL_SMILES")

    if mole_fraction_col is not None:
        if mole_fraction_col not in data.columns:
            raise ValueError(
                f"Mole fraction column '{mole_fraction_col}' not found in input data. "
                f"Available columns: {list(data.columns)}"
            )
        data.rename(columns={mole_fraction_col: "Mole_fraction"}, inplace=True)

    if temp_col is None or temp_col not in data.columns:
        data["Temperature"] = default_temp
    elif data[temp_col].isna().any():
        data["Temperature"] = data[temp_col].fillna(default_temp)
    else:
        data.rename(columns={temp_col: "Temperature"}, inplace=True)

    return data


def predict(
    data: pd.DataFrame,
    ensemble: LoadedEnsemble,
    smiles_col: str = "Standardized_IL_SMILES",
    temp_col: str = "Temperature",
    mole_fraction_col: str | None = "Mole_fraction",
) -> pd.DataFrame:
    """Run the 5-model ensemble over already-prepared ``data`` and append prediction columns.

    ``mole_fraction_col=None`` skips the mole-fraction feature (pure-IL models). If an
    individual ensemble member fails, its predictions are recorded as NaN so the remaining
    models can still contribute to the consensus.
    """
    data = data.copy()
    all_smiles_list = data[smiles_col].tolist()
    predictions = []

    for i, (model, metadata) in enumerate(zip(ensemble.models, ensemble.metadata), 1):
        try:
            required_descriptors_model = metadata["descriptors"]
            descriptor_array = process_il_smiles_list(all_smiles_list, required_descriptors_model)
            temperature_array = data[temp_col].values.reshape(-1, 1)
            feature_arrays = [descriptor_array, temperature_array]
            if mole_fraction_col is not None:
                feature_arrays.append(data[mole_fraction_col].values.reshape(-1, 1))
            descriptor_array = np.hstack(feature_arrays)
        except Exception as e:
            print(f"Error calculating descriptors for Model {i}: {e}")
            predictions.append(pd.Series([np.nan] * len(all_smiles_list)))
            continue

        try:
            preds = model.predict(descriptor_array)
            predictions.append(pd.Series(preds))
        except Exception as e:
            print(f"Error processing Model {i}: {e}")
            predictions.append(pd.Series([np.nan] * len(all_smiles_list)))

    data["prediction_mean"] = pd.concat(predictions, axis=1).mean(axis=1).round(4)
    data["prediction_std"] = pd.concat(predictions, axis=1).std(axis=1).round(4)
    return data


def run_prediction(
    data: pd.DataFrame,
    spec: "ModelSpec",
    ensemble: LoadedEnsemble | None = None,
    smiles_col: str | None = None,
    temp_col: str | None = None,
    mole_fraction_col: str | None = None,
) -> pd.DataFrame:
    """Single in-process entry point used by the CLI, the Streamlit app, and tests.

    Any column argument left as ``None`` falls back to ``spec``'s defaults. Pass an
    already-loaded ``ensemble`` (e.g. from a cache) to avoid reloading the joblib files.
    """
    smiles_col = smiles_col or spec.default_smiles_col
    temp_col = temp_col or spec.default_temp_col
    if spec.is_pure:
        mole_fraction_col = None
    else:
        mole_fraction_col = mole_fraction_col or spec.default_mole_fraction_col

    prepared = prepare_input(
        data,
        smiles_col=smiles_col,
        temp_col=temp_col,
        mole_fraction_col=mole_fraction_col,
        default_temp=DEFAULT_TEMPERATURE_K,
    )

    resolved_mole_fraction_col = "Mole_fraction" if mole_fraction_col is not None else None

    if ensemble is None:
        ensemble = load_models_and_metadata(spec.model_dir)

    return predict(
        prepared,
        ensemble,
        smiles_col="Standardized_IL_SMILES",
        temp_col="Temperature",
        mole_fraction_col=resolved_mole_fraction_col,
    )
