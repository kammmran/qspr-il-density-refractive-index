Function Guide
==============

The eight ``apply_*.py`` files use the same prediction functions. The example
code below is taken from ``models/ri_ethanol/apply_ri_ethanol.py``; density,
other-solvent, and pure-ionic-liquid scripts use the same structure with
property-specific defaults and feature columns.

1. ``standardize_molecule``
---------------------------

**Purpose:** validate and normalize one input SMILES string before descriptors
are calculated.

**Input:** ``smi``, a SMILES string. Ionic liquids can contain multiple
components separated by ``.``.

**Output:** a tuple containing the standardized SMILES and a human-readable
change summary. Invalid input returns the original string and an error label.

The function performs three transformations in order:

* Reionizes each component and rebuilds the dot-separated SMILES.
* Applies RDKit functional-group normalization.
* Removes stereochemical information so equivalent structures use a consistent
  descriptor representation.

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :lines: 26-73
   :linenos:

2. ``reorder_charged_species``
------------------------------

**Purpose:** make the component order deterministic for ionic liquids.

**Input:** a pandas DataFrame and the name of the SMILES column.

**Output:** the same DataFrame after each SMILES value has been rewritten with
cation components first and anion components second. Uncharged components are
not added to either list.

The nested ``reorder_smiles`` function handles one row, while ``DataFrame.apply``
applies it to the complete column.

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :lines: 75-90
   :linenos:

3. ``load_models_and_metadata``
-------------------------------

**Purpose:** load the five members of an ensemble and the feature description
needed by each member.

**Input:** ``model_dir``, the directory containing ``model_1.joblib`` through
``model_5.joblib`` and matching ``metadata_*.json`` files.

**Output:** two parallel lists: loaded estimators and metadata dictionaries.

The function checks every expected file before loading it. A missing file raises
``FileNotFoundError`` instead of allowing a partial ensemble to run.

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :lines: 91-107
   :linenos:

4. ``calculate_descriptors``
----------------------------

**Purpose:** convert one ionic-liquid SMILES string into the feature vector
expected by one trained model.

**Input:** ``smiles`` and that model's ``required_descriptors`` list.

**Output:** a mean descriptor vector and descriptor names, or ``(None, None)``
when no component can be parsed.

Mordred is configured without 3D descriptors. For a multi-component SMILES,
descriptors are calculated for each valid component and averaged column by
column. Mordred ``Missing`` values become ``numpy.nan`` so the averaging step
can ignore missing values.

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :lines: 108-141
   :linenos:

5. ``process_il_smiles_list``
-----------------------------

**Purpose:** build the descriptor matrix for every row in the input CSV.

**Input:** a list of standardized SMILES strings and one model's descriptor
list.

**Output:** a two-dimensional NumPy array with one row per input molecule.

If a row has no valid descriptors, the function inserts a NaN-filled row with
the expected descriptor count. This keeps the matrix shape compatible with the
model and lets the later prediction step report unusable rows as NaN.

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :lines: 142-153
   :linenos:

6. ``main``
-----------

**Purpose:** coordinate the complete prediction workflow.

The function:

1. Applies defaults to optional command-line arguments.
2. Verifies the input path and required CSV columns.
3. Standardizes and reorders every SMILES value.
4. Creates a default temperature column when needed and fills missing
   temperatures with 298.15 K.
5. Loads the five estimators and calculates model-specific descriptor matrices.
6. Appends temperature and mole fraction features for mixture models.
7. Calculates ensemble mean and standard deviation, rounds them to four decimal
   places, and writes the result CSV.

Each model is evaluated independently. If one model fails, its prediction row
is replaced with NaN so the remaining ensemble members can still be combined.

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :lines: 154-245
   :linenos:

Pure ionic-liquid scripts use the same ``main`` workflow but do not require a
mole-fraction column and append only temperature to the descriptor matrix.

7. ``configure_prediction_args``
--------------------------------

**Purpose:** provide interactive settings when a script is started without an
input path.

**Input:** an argparse namespace plus the model's default directory and output
filename.

**Output:** the same namespace with input, column, model, and output settings
filled in. Existing command-line values are preserved when ``input_csv`` was
provided.

Blank optional answers keep their displayed defaults. The input path is the
only mandatory interactive answer.

.. literalinclude:: ../models/prediction_settings.py
   :language: python
   :lines: 4-29
   :linenos:

8. ``_prompt``
--------------

**Purpose:** implement one consistent prompt with optional default handling.

When a default is available, it is shown in brackets and returned when the user
presses Enter. For prompts without a default, the stripped user input is
returned unchanged.

.. literalinclude:: ../models/prediction_settings.py
   :language: python
   :lines: 30-34
   :linenos: