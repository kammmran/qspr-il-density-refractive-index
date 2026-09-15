"""Streamlit UI wiring tests: the "Fetch & clean data" page must pass the system the user
picked (pure / binary / ternary / all) through to fetch_curated_dataset.

Driven headlessly through streamlit's own AppTest harness, so these exercise the real
widgets and callbacks rather than re-implementing the page's logic.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="GUI extra not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

from qspr_il import app  # noqa: E402

APP_PATH = str(Path(app.__file__).resolve())
PROPERTY_SELECTBOX, SYSTEM_SELECTBOX = 0, 1


@pytest.fixture
def fetch_calls():
    """Patch out the network fetch, recording how the page called it."""
    calls = []

    def fake_fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame(
            {
                "IL_SMILES": ["CCCCn1cc[n+](C)c1.F[B-](F)(F)F"],
                "Temperature (K)": [298.15],
                "Property": ["Density"],
                "Property_value": [1050.0],
                "Mole_fraction_IL": [0.4],
            }
        )

    with patch("qspr_il.data.cleaning.fetch_curated_dataset", fake_fetch):
        yield calls


def _fetch_page(system_type, solvents=(), limit_temperature=True):
    """Render the fetch page, choose a system (and solvents), then click Fetch."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Fetch & clean data").run()
    at.selectbox[SYSTEM_SELECTBOX].set_value(system_type).run()
    for key, value in solvents:
        at.selectbox(key=key).set_value(value).run()
    if not limit_temperature:
        at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def test_pure_system_passes_no_solvents(fetch_calls):
    _fetch_page(app.SYSTEM_PURE)
    (_property, solvent_name, solvent_name_2), kwargs = fetch_calls[0]
    assert solvent_name is None
    assert solvent_name_2 is None
    assert kwargs["all_systems"] is False


def test_all_systems_option_sets_the_all_systems_flag(fetch_calls):
    at = _fetch_page(app.SYSTEM_ALL)
    (_property, solvent_name, solvent_name_2), kwargs = fetch_calls[0]
    assert solvent_name is None
    assert solvent_name_2 is None
    assert kwargs["all_systems"] is True
    # The auto-detection caveat matters here: mixtures whose solvent isn't one this pipeline
    # recognizes get skipped, so the page has to say so.
    assert any("told apart automatically" in c.value for c in at.caption)


def test_binary_system_passes_one_solvent(fetch_calls):
    _fetch_page(app.SYSTEM_BINARY, [("solvent1_choice", "ethanol")])
    (_property, solvent_name, solvent_name_2), kwargs = fetch_calls[0]
    assert solvent_name == "ethanol"
    assert solvent_name_2 is None
    assert kwargs["all_systems"] is False


def test_ternary_system_passes_both_solvents(fetch_calls):
    _fetch_page(
        app.SYSTEM_TERNARY,
        [("solvent1_choice", "ethanol"), ("solvent2_choice", "water")],
    )
    (_property, solvent_name, solvent_name_2), kwargs = fetch_calls[0]
    assert solvent_name == "ethanol"
    assert solvent_name_2 == "water"
    assert kwargs["all_systems"] is False


def test_unlimited_temperature_passes_no_temperature_range(fetch_calls):
    _fetch_page(app.SYSTEM_PURE, limit_temperature=False)
    _args, kwargs = fetch_calls[0]
    assert kwargs["temp_range"] is None
    assert kwargs["pressure_range"] == (90.0, 110.0)


def test_ternary_fetch_without_a_matching_model_offers_auto_training(fetch_calls):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Both").run()
    at.selectbox[SYSTEM_SELECTBOX].set_value(app.SYSTEM_TERNARY).run()
    at.selectbox(key="solvent1_choice").set_value("ethanol").run()
    at.selectbox(key="solvent2_choice").set_value("water").run()
    at.button[0].click().run()
    assert not at.exception, at.exception

    # No ternary model is registered, so the page must fall through to auto-training, with
    # both solvents reflected in the suggested model directory.
    assert any("No trained prediction model exists yet" in i.value for i in at.info)
    assert any("ethanol_water_ensemble_model" in ti.value for ti in at.text_input)
