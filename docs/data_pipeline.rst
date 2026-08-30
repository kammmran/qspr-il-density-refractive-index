Data Acquisition and Cleaning
==============================

Training datasets are sourced from the `ILThermo database
<https://ilthermo.boulder.nist.gov/>`_. This project used to depend on an
external tool, `pyIonics <https://github.com/kammmran/pyionics>`_, for that
step; pyIonics is being deprecated as a standalone package, so its
functionality is now vendored permanently inside :mod:`qspr_il.data.ionics`.
A second module, :mod:`qspr_il.data.cleaning`, adds the "duplicate removal,
consistency checks, and standardization" step the project has always
described but never actually implemented in code until now.

Fetch layer: ``qspr_il.data.ionics``
-------------------------------------

Pure fetch/reshape -- no cleaning logic. Downloads raw measurement data from
ILThermo, flattens it into CSV/TSV, and joins in SMILES strings for known
compound ids.

.. automodule:: qspr_il.data.ionics.client
   :members:
   :undoc-members:
   :show-inheritance:

Cleaning layer: ``qspr_il.data.cleaning``
-------------------------------------------

Deduplication, temperature/pressure filtering, SMILES standardization, and
column normalization into the same shape as ``datasets/training_sets/*.csv``.

.. automodule:: qspr_il.data.cleaning
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

To (re)build a curated CSV for one property/solvent combination::

   from qspr_il.data.cleaning import fetch_curated_dataset

   df = fetch_curated_dataset("dens", "ethanol")   # density, IL in ethanol
   df = fetch_curated_dataset("n", None)            # refractive index, pure IL

``property_short`` is ``"dens"`` (density) or ``"n"`` (refractive index);
``solvent_name`` is one of ``"water"``, ``"ethanol"``, ``"isopropanol"``, or
``None`` for a pure ionic liquid. Pass ``pressure_range``, ``temp_range``, or
``property_range`` to narrow the accepted conditions -- for example, to
restrict to strictly atmospheric pressure or a tighter temperature window.

This pipeline only fetches and cleans data; it does not train or refit any
model. The existing trained ``.joblib`` ensembles under
``qspr_il/models/<name>/`` are unaffected and keep being used for prediction.

Known data-source limitations
------------------------------

Two staleness issues were found (empirically, against the live API) while
building this pipeline:

* The bundled ``property_idsets.csv`` lookup table's internal ILThermo
  property ids no longer match what the live search API expects.
  :func:`qspr_il.data.cleaning.fetch_curated_dataset` works around this by
  searching broadly and filtering client-side on the exact property display
  name instead.
* The bundled ``smiles.csv`` compound-id lookup table has sparse-to-nonexistent
  coverage of actual ionic-liquid compounds (it appears to mostly cover common
  small molecules/solvents). Solvent SMILES are therefore resolved by name
  (:data:`qspr_il.data.cleaning.SOLVENT_SMILES_BY_NAME`) rather than by id.
  IL-side rows whose SMILES can't be resolved are dropped rather than fed into
  a model -- refreshing ``smiles.csv`` from a current ILThermo compound export
  would recover more IL coverage, but is out of scope for this pipeline.
