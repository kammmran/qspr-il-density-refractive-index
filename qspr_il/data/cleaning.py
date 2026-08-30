"""Fetch + clean pipeline: turns raw ILThermo data into curated, training-ready CSVs.

Nothing in this repo previously implemented the "duplicate removal, consistency
checks, and standardization" the README describes — :mod:`qspr_il.data.ionics`
only fetches and flattens raw data. This module is that missing piece. It does
**not** do any model training/fitting — its output is a curated CSV shaped like
the existing ``datasets/training_sets/*.csv`` files, for a human to use however
they choose (including, but not limited to, retraining models outside this
pipeline's scope).

Column semantics here were reverse-engineered from real, live ILThermo API
responses (not just the existing curated CSVs) since no raw sample was
committed to the repo. In particular:

* The bundled ``keydata/property_idsets.csv`` "prp" ids used by
  :func:`qspr_il.data.ionics.client.getIdsets` were found to return zero
  results against the *current* live API (verified empirically) -- ILThermo's
  internal ids appear to have drifted since that lookup table was built. This
  module works around that by searching broadly (unfiltered by ``prp``) and
  then filtering the returned idsets client-side by their exact property
  display name (e.g. ``"Density"``, ``"Refractive index"``), which was
  confirmed to still work reliably.
* The bundled ``keydata/smiles.csv`` compound-id lookup table is similarly a
  point-in-time snapshot and may not cover every compound id ILThermo returns
  today; :func:`qspr_il.data.ionics.client.addSmiles` silently falls back to
  the raw molecular formula string when a compound id isn't found. This
  module treats that as an expected case: any component whose "SMILES" field
  fails to parse via RDKit is dropped rather than fed into a model.
* **Known limitation, confirmed empirically:** live spot-checks against
  ``https://ilthermo.boulder.nist.gov`` found the bundled table's IL-compound
  coverage is sparse to nonexistent (0 of several dozen freshly-downloaded IL
  component ids resolved, across both pure and mixture density datasets) --
  the table appears to mostly cover common small molecules/solvents rather
  than the complex IL cations/anions this project models. Even solvent
  component *ids* drift from what's in the table (though the table's SMILES
  for that solvent *name* is correct), which is why solvent SMILES are
  resolved by name via :data:`SOLVENT_SMILES_BY_NAME` instead of by id. Until
  ``keydata/smiles.csv`` is refreshed from a current ILThermo compound export,
  :func:`fetch_curated_dataset` will legitimately return few or no IL rows for
  most searches -- this is the pipeline correctly refusing to fabricate
  structures, not a bug in the filtering logic itself.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles

from qspr_il.data.ionics import client as ionics_client
from qspr_il.models.engine import reorder_charged_species, standardize_molecule

# Elements whose presence marks an ionic liquid as metal-containing (alkali, alkaline-earth,
# transition, and post-transition metals commonly seen in IL literature).
METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

# ILThermo's exact "property" display names for the two properties this project cares about.
PROPERTY_DISPLAY_NAMES = {
    "dens": "Density",
    "n": "Refractive index",
}

# Solvents this project models explicitly; anything else in a binary system is treated as the IL.
KNOWN_SOLVENTS = {"water", "ethanol", "isopropanol", "2-propanol"}

# The live ILThermo API was found (empirically) to assign a component id for a given named
# solvent that does not always match the id addSmiles's bundled keydata/smiles.csv table used
# for that same compound name -- even though the table does contain the right SMILES under a
# different id. Since the solvents this project cares about are a small, fixed set, resolving
# them by name here is far more reliable than depending on id drift in the fetched data.
SOLVENT_SMILES_BY_NAME = {
    "water": "O",
    "ethanol": "CCO",
    "isopropanol": "CC(C)O",
    "2-propanol": "CC(C)O",
}

MIXTURE_COLUMNS = {
    "dens": [
        "setid", "Temperature (K)", "Pressure (kPa)", "Density (kg/m3)", "reference", "phases",
        "Mole fraction", "Mole fraction of compound name", "Weight fraction", "Weight fraction of compound name",
        "IL_SMILES", "IL_name", "IL_ID", "Solvent_SMILES", "Solvent_name", "Solvent_ID",
        "IL_cation_SMILES", "IL_anion_SMILES", "IL_component_number", "Mole fraction of solvent",
        "IL_MW", "Solvent_MW", "Record_ID", "Contains_Metal", "Standardized_IL_SMILES", "Changes",
    ],
    "n": [
        "setid", "Temperature (K)", "Pressure (kPa)", "Wavelength (nm)", "Refractive index (Na D-line)",
        "reference", "phases", "Mole fraction", "Compound (mole fraction)", "Weight fraction",
        "Compound (weight fraction)", "IL_SMILES", "IL_name", "IL_ID", "Solvent_SMILES", "Solvent_name",
        "Solvent_ID", "IL_cation_SMILES", "IL_anion_SMILES", "IL_component_number", "Mole fraction of solvent",
        "IL_MW", "Solvent_MW", "Record_ID", "Contains_Metal", "Standardized_IL_SMILES", "Changes",
    ],
}
PURE_COLUMNS = {
    "dens": [
        "setid", "Temperature (K)", "Pressure (kPa)", "Density (kg/m3)", "reference", "phases",
        "IL_ID", "IL_SMILES", "IL_name", "IL_cation_SMILES", "IL_anion_SMILES", "IL_component_number",
        "Record_ID", "Contains_Metal", "Standardized_IL_SMILES", "Changes",
    ],
    "n": [
        "setid", "Temperature (K)", "Pressure (kPa)", "Wavelength (nm)", "Refractive index (Na D-line)",
        "reference", "phases", "IL_ID", "IL_SMILES", "IL_name", "IL_cation_SMILES", "IL_anion_SMILES",
        "IL_component_number", "Record_ID", "Contains_Metal", "Standardized_IL_SMILES", "Changes",
    ],
}

_TARGET_COLUMN = {"dens": "Density (kg/m3)", "n": "Refractive index (Na D-line)"}


def compute_molecular_fields(smiles: str) -> dict:
    """Standardize a (possibly multi-component) IL SMILES and derive descriptive fields.

    Returns a dict with ``Standardized_IL_SMILES``, ``Changes``, ``IL_cation_SMILES``,
    ``IL_anion_SMILES``, ``IL_component_number``, ``IL_MW``, and ``Contains_Metal``. If the
    SMILES fails to parse, ``Standardized_IL_SMILES`` is the original input and ``IL_MW`` is NaN.
    """
    standardized, changes = standardize_molecule(smiles)
    if changes == "Invalid SMILES":
        return {
            "Standardized_IL_SMILES": standardized,
            "Changes": changes,
            "IL_cation_SMILES": "",
            "IL_anion_SMILES": "",
            "IL_component_number": 0,
            "IL_MW": np.nan,
            "Contains_Metal": False,
        }

    df = pd.DataFrame({"Standardized_IL_SMILES": [standardized]})
    reorder_charged_species(df, smiles_col="Standardized_IL_SMILES")
    standardized = df.loc[0, "Standardized_IL_SMILES"]

    parts = standardized.split(".")
    cations = [p for p in parts if re.search(r"\+[0-9]*", p)]
    anions = [p for p in parts if re.search(r"\-[0-9]*", p)]

    mol = MolFromSmiles(standardized)
    mw = Descriptors.MolWt(mol) if mol is not None else np.nan
    contains_metal = any(atom.GetSymbol() in METALS for atom in mol.GetAtoms()) if mol is not None else False

    return {
        "Standardized_IL_SMILES": standardized,
        "Changes": changes,
        "IL_cation_SMILES": ".".join(cations),
        "IL_anion_SMILES": ".".join(anions),
        "IL_component_number": len(parts),
        "IL_MW": mw,
        "Contains_Metal": contains_metal,
    }


def flag_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
    conflict_tolerance_frac: float = 0.02,
) -> pd.Series:
    """Flag rows as ``unique`` / ``duplicate_dropped`` / ``conflicting_duplicate``.

    Rows sharing the same ``key_cols`` are grouped. Within a group, if every ``value_col``
    reading agrees within ``conflict_tolerance_frac`` (relative) of the group median, the
    first row is kept as ``unique`` and the rest flagged ``duplicate_dropped``. Otherwise
    every row in the group is flagged ``conflicting_duplicate`` -- conflicting measurements
    are surfaced for a human to resolve, not silently averaged away.
    """
    flags = pd.Series("unique", index=df.index, dtype=object)
    for _, group in df.groupby(key_cols, dropna=False):
        if len(group) == 1:
            continue
        values = group[value_col].astype(float)
        median = values.median()
        if median == 0 or median != median:  # zero or NaN median: fall back to absolute tolerance
            within_tolerance = (values - median).abs() <= conflict_tolerance_frac
        else:
            within_tolerance = ((values - median).abs() / abs(median)) <= conflict_tolerance_frac
        if within_tolerance.all():
            idxs = list(group.index)
            flags.loc[idxs[1:]] = "duplicate_dropped"
        else:
            flags.loc[group.index] = "conflicting_duplicate"
    return flags


def filter_conditions(
    df: pd.DataFrame,
    pressure_col: str = "Pressure (kPa)",
    pressure_range: tuple[float, float] = (90.0, 110.0),
    allow_missing_pressure: bool = True,
    temp_col: str = "Temperature (K)",
    temp_range: tuple[float, float] = (253.0, 573.0),
) -> pd.DataFrame:
    """Keep only rows within the given pressure/temperature ranges.

    ``pressure_range`` defaults to the project's documented near-atmospheric scope
    (90-110 kPa). Both ranges are parameters, not hardcoded constants, so callers can widen
    or narrow them per property/solvent as needed.
    """
    mask = df[temp_col].between(*temp_range)
    if pressure_col in df.columns:
        pressure_mask = df[pressure_col].between(*pressure_range)
        if allow_missing_pressure:
            pressure_mask = pressure_mask | df[pressure_col].isna()
        mask = mask & pressure_mask
    return df[mask].copy()


def filter_property_range(df: pd.DataFrame, value_col: str, plausible_range: tuple[float, float]) -> pd.DataFrame:
    """Keep only rows whose ``value_col`` falls within a caller-supplied physical sanity range."""
    return df[df[value_col].between(*plausible_range)].copy()


def _find_column(columns, pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for col in columns:
        if regex.search(col):
            return col
    return None


def _component_indices(columns) -> list[int]:
    indices = set()
    for col in columns:
        m = re.match(r"component (\d+) (idout|name|SMILES)$", col)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def build_curated_dataset(
    raw_df: pd.DataFrame,
    property_short: str,
    solvent_name: str | None,
    pressure_range: tuple[float, float] = (90.0, 110.0),
    temp_range: tuple[float, float] = (253.0, 573.0),
    property_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Turn a raw, flattened+SMILES-joined ILThermo CSV (concatenated across idsets) into a
    curated, training-set-shaped DataFrame.

    ``raw_df`` is the concatenation of one or more CSVs produced by
    ``ionics.convert2csv`` + ``ionics.addSmiles``. ``property_short`` is ``"dens"`` or
    ``"n"``; ``solvent_name`` is one of :data:`KNOWN_SOLVENTS`, or ``None`` for a pure-IL
    dataset.
    """
    is_pure = solvent_name is None
    columns = raw_df.columns
    temp_col = _find_column(columns, r"^Temperature")
    pressure_col = _find_column(columns, r"^Pressure")
    property_col = _find_column(columns, PROPERTY_DISPLAY_NAMES[property_short].split()[0])
    if temp_col is None or property_col is None:
        raise ValueError("raw_df is missing a recognizable Temperature or property column.")

    mole_fraction_col = _find_column(columns, r"^Mole fraction of ")
    component_idxs = _component_indices(columns)

    rows = []
    for _, row in raw_df.iterrows():
        components = []
        for idx in component_idxs:
            name_col, smiles_col, idout_col = (
                f"component {idx} name",
                f"component {idx} SMILES",
                f"component {idx} idout",
            )
            if name_col not in row or smiles_col not in row:
                continue
            components.append(
                {"idx": idx, "name": row[name_col], "smiles": row[smiles_col], "idout": row.get(idout_col, "")}
            )

        if is_pure:
            if len(components) != 1:
                continue
            il_component = components[0]
            solvent_component = None
        else:
            solvent_component = next(
                (c for c in components if str(c["name"]).strip().lower() == solvent_name.lower()), None
            )
            il_component = next((c for c in components if c is not solvent_component), None)
            if solvent_component is None or il_component is None:
                continue
            # Resolve the solvent's SMILES by name (see SOLVENT_SMILES_BY_NAME) rather than
            # trusting addSmiles's id-based lookup, which was found to miss known solvents
            # when ILThermo's live component id doesn't match the bundled keydata table's id
            # for that same compound.
            known_solvent_smiles = SOLVENT_SMILES_BY_NAME.get(str(solvent_component["name"]).strip().lower())
            if known_solvent_smiles:
                solvent_component = {**solvent_component, "smiles": known_solvent_smiles}

        molecular_fields = compute_molecular_fields(il_component["smiles"])
        if molecular_fields["Changes"] == "Invalid SMILES":
            continue  # unparseable SMILES (e.g. a stale keydata lookup fell back to a formula)

        record = {
            "setid": row.get("idset", ""),
            "Temperature (K)": row.get(temp_col),
            "Pressure (kPa)": row.get(pressure_col) if pressure_col else np.nan,
            _TARGET_COLUMN[property_short]: row.get(property_col),
            "reference": row.get("author", ""),
            "phases": (property_col.split(" - ")[-1] if " - " in property_col else ""),
            "IL_ID": il_component["idout"],
            "IL_SMILES": il_component["smiles"],
            "IL_name": il_component["name"],
            **molecular_fields,
        }

        if not is_pure:
            fraction_value = row.get(mole_fraction_col) if mole_fraction_col else np.nan
            fraction_component_name = (
                re.sub(r"^Mole fraction of ", "", mole_fraction_col, flags=re.IGNORECASE) if mole_fraction_col else ""
            )
            solvent_mw_mol = MolFromSmiles(solvent_component["smiles"])
            record.update(
                {
                    "Solvent_ID": solvent_component["idout"],
                    "Solvent_SMILES": solvent_component["smiles"],
                    "Solvent_name": solvent_component["name"],
                    "Solvent_MW": Descriptors.MolWt(solvent_mw_mol) if solvent_mw_mol is not None else np.nan,
                    "Mole fraction": fraction_value,
                    "Weight fraction": np.nan,
                }
            )
            if property_short == "dens":
                record["Mole fraction of compound name"] = fraction_component_name
                record["Weight fraction of compound name"] = np.nan
            else:
                record["Compound (mole fraction)"] = fraction_component_name
                record["Compound (weight fraction)"] = np.nan
            if fraction_component_name.strip().lower() == solvent_name.lower() and fraction_value == fraction_value:
                record["Mole fraction of solvent"] = fraction_value
            elif fraction_value == fraction_value:
                record["Mole fraction of solvent"] = 1.0 - fraction_value
            else:
                record["Mole fraction of solvent"] = np.nan

        rows.append(record)

    target_columns = PURE_COLUMNS[property_short] if is_pure else MIXTURE_COLUMNS[property_short]
    curated = pd.DataFrame(rows)
    if curated.empty:
        return pd.DataFrame(columns=target_columns)

    curated["Record_ID"] = [f"{r['setid']}_{i}" for i, r in enumerate(rows)]
    curated = filter_conditions(curated, pressure_range=pressure_range, temp_range=temp_range)
    if property_range is not None:
        curated = filter_property_range(curated, _TARGET_COLUMN[property_short], property_range)

    key_cols = ["IL_SMILES", "Temperature (K)"] if is_pure else ["IL_SMILES", "Solvent_name", "Temperature (K)"]
    curated["Data_quality_flag"] = flag_duplicates(curated, key_cols, _TARGET_COLUMN[property_short])
    curated = curated[curated["Data_quality_flag"] != "duplicate_dropped"]

    for col in target_columns:
        if col not in curated.columns:
            curated[col] = np.nan
    return curated.reindex(columns=target_columns).reset_index(drop=True)


def fetch_curated_dataset(
    property_short: str,
    solvent_name: str | None,
    data_root=None,
    max_datasets: int | None = None,
    **filter_kwargs,
) -> pd.DataFrame:
    """Download, flatten, SMILES-join, and clean ILThermo data for one property/solvent combo.

    This is the single high-level function a user calls to (re)generate a curated CSV, e.g.
    ``fetch_curated_dataset("dens", "ethanol")`` to rebuild something shaped like
    ``datasets/training_sets/density_ethanol.csv``. Pass ``max_datasets`` to cap how many
    ILThermo datasets are downloaded (useful for quick, rate-limit-friendly test runs).
    """
    ncmp = "1" if solvent_name is None else "2"
    data_root_path = ionics_client.get_data_dir(data_root)

    idsets_path = ionics_client.getIdsets(prop=None, ncmp=ncmp, data_root=data_root_path)
    idsets_json = json.load(open(idsets_path))
    display_name = PROPERTY_DISPLAY_NAMES[property_short]
    matching = [row[0] for row in idsets_json.get("res", []) if row[2] == display_name]
    if max_datasets is not None:
        matching = matching[:max_datasets]

    folder_name = f"{property_short}_{'pure' if solvent_name is None else solvent_name}_data"
    ionics_client.download_idsets(matching, output_dir=data_root_path / folder_name)
    ionics_client.convert2csv(folder_name=folder_name, data_root=data_root_path)
    ionics_client.addSmiles(folder_name=f"csv_{folder_name}", data_root=data_root_path)

    smiles_dir = data_root_path / f"smiles_csv_{folder_name}"
    frames = [pd.read_csv(f) for f in smiles_dir.glob("*.csv")] if smiles_dir.exists() else []
    if not frames:
        target_columns = PURE_COLUMNS[property_short] if solvent_name is None else MIXTURE_COLUMNS[property_short]
        return pd.DataFrame(columns=target_columns)

    raw_df = pd.concat(frames, ignore_index=True)
    return build_curated_dataset(raw_df, property_short, solvent_name, **filter_kwargs)
