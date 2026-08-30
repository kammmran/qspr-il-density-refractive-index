import numpy as np
import pandas as pd

from qspr_il.data import cleaning

# A real, valid SMILES for an imidazolium-based IL and its cation/anion parts,
# used across tests so RDKit standardization succeeds.
IL_SMILES = "CCCCn1cc[n+](C)c1.F[B-](F)(F)F"
ETHANOL_SMILES = "CCO"


def _pure_raw_row(**overrides):
    row = {
        "idset": "SET1",
        "author": "Doe et al.",
        "Temperature, K": 298.15,
        "Pressure, kPa": 101.0,
        "Specific density, kg/m3 - Liquid": 1050.0,
        "component 1 idout": "C001",
        "component 1 name": "1-butyl-3-methylimidazolium tetrafluoroborate",
        "component 1 SMILES": IL_SMILES,
    }
    row.update(overrides)
    return row


def _mixture_raw_row(**overrides):
    row = {
        "idset": "SET2",
        "author": "Doe et al.",
        "Temperature, K": 298.15,
        "Pressure, kPa": 101.0,
        "Mole fraction of ethanol": 0.4,
        "Specific density, kg/m3 - Liquid": 900.0,
        "component 1 idout": "C002",
        "component 1 name": "ethanol",
        "component 1 SMILES": ETHANOL_SMILES,
        "component 2 idout": "C001",
        "component 2 name": "1-butyl-3-methylimidazolium tetrafluoroborate",
        "component 2 SMILES": IL_SMILES,
    }
    row.update(overrides)
    return row


def test_compute_molecular_fields_valid_smiles():
    fields = cleaning.compute_molecular_fields(IL_SMILES)
    assert fields["IL_component_number"] == 2
    assert fields["IL_cation_SMILES"]
    assert fields["IL_anion_SMILES"]
    assert fields["IL_MW"] > 0
    assert fields["Contains_Metal"] is False


def test_compute_molecular_fields_detects_metal():
    # Sodium acetate-like fragment: contains a metal cation.
    fields = cleaning.compute_molecular_fields("[Na+].CC(=O)[O-]")
    assert fields["Contains_Metal"] is True


def test_flag_duplicates_exact_duplicate_dropped():
    df = pd.DataFrame(
        {
            "IL_SMILES": ["A", "A", "B"],
            "Temperature (K)": [298.15, 298.15, 300.0],
            "Density (kg/m3)": [1000.0, 1000.0, 900.0],
        }
    )
    flags = cleaning.flag_duplicates(df, ["IL_SMILES", "Temperature (K)"], "Density (kg/m3)")
    assert flags.tolist() == ["unique", "duplicate_dropped", "unique"]


def test_flag_duplicates_conflicting_values_flagged():
    df = pd.DataFrame(
        {
            "IL_SMILES": ["A", "A"],
            "Temperature (K)": [298.15, 298.15],
            "Density (kg/m3)": [1000.0, 1200.0],  # >2% apart
        }
    )
    flags = cleaning.flag_duplicates(df, ["IL_SMILES", "Temperature (K)"], "Density (kg/m3)")
    assert (flags == "conflicting_duplicate").all()


def test_filter_conditions_drops_out_of_range_pressure():
    df = pd.DataFrame(
        {
            "Temperature (K)": [298.15, 298.15, 298.15],
            "Pressure (kPa)": [101.0, 500.0, np.nan],
        }
    )
    filtered = cleaning.filter_conditions(df, allow_missing_pressure=True)
    assert len(filtered) == 2  # 101.0 kept, 500.0 dropped, NaN kept


def test_filter_property_range():
    df = pd.DataFrame({"Density (kg/m3)": [500.0, 5000.0, 1200.0]})
    filtered = cleaning.filter_property_range(df, "Density (kg/m3)", (300.0, 3000.0))
    assert filtered["Density (kg/m3)"].tolist() == [500.0, 1200.0]


def test_build_curated_dataset_pure_columns_match_target_schema():
    raw_df = pd.DataFrame([_pure_raw_row()])
    curated = cleaning.build_curated_dataset(raw_df, "dens", solvent_name=None)
    assert list(curated.columns) == cleaning.PURE_COLUMNS["dens"]
    assert len(curated) == 1
    assert curated.loc[0, "Density (kg/m3)"] == 1050.0
    assert curated.loc[0, "phases"] == "Liquid"


def test_build_curated_dataset_mixture_columns_match_target_schema():
    raw_df = pd.DataFrame([_mixture_raw_row()])
    curated = cleaning.build_curated_dataset(raw_df, "dens", solvent_name="ethanol")
    assert list(curated.columns) == cleaning.MIXTURE_COLUMNS["dens"]
    assert len(curated) == 1
    assert curated.loc[0, "Solvent_name"] == "ethanol"
    assert curated.loc[0, "IL_name"] == "1-butyl-3-methylimidazolium tetrafluoroborate"


def test_build_curated_dataset_drops_unparseable_smiles():
    raw_df = pd.DataFrame([_pure_raw_row(**{"component 1 SMILES": "C12H19F6N3O4S2"})])  # a bare formula, not SMILES
    curated = cleaning.build_curated_dataset(raw_df, "dens", solvent_name=None)
    assert curated.empty


def test_build_curated_dataset_filters_by_temperature_range():
    raw_df = pd.DataFrame([_pure_raw_row(**{"Temperature, K": 1000.0})])
    curated = cleaning.build_curated_dataset(raw_df, "dens", solvent_name=None, temp_range=(253.0, 573.0))
    assert curated.empty
