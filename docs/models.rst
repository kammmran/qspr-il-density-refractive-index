Model Application Files
=======================

The model application files below load the trained five-model ensembles,
calculate molecular descriptors, and write prediction means and standard
deviations to CSV. The trained ``.joblib`` files and their JSON metadata remain
in the repository under ``models/`` as runtime artifacts.

Prediction pipeline
-------------------

Each ``apply_*.py`` script follows the same sequence:

1. ``main(args)`` reads the CSV and applies defaults for optional settings.
2. ``standardize_molecule(smi)`` validates and normalizes each SMILES string,
   including reionization, functional-group normalization, and stereo cleanup.
3. ``reorder_charged_species(df)`` places cation components before anion
   components in multi-component ionic-liquid SMILES strings.
4. ``load_models_and_metadata(model_dir)`` loads the five ensemble models and
   their descriptor metadata files.
5. ``calculate_descriptors(smiles, required_descriptors)`` calculates the
   Mordred descriptors required by one model and averages descriptors across
   components of a multi-component SMILES string.
6. ``process_il_smiles_list(smiles_list, required_descriptors)`` repeats the
   descriptor calculation for every input row and creates the feature matrix.
7. ``main(args)`` appends temperature and, for mixture models, mole fraction;
   it then combines the five predictions into ``prediction_mean`` and
   ``prediction_std`` columns.

Function reference
------------------

``standardize_molecule(smi)``
   Returns a standardized SMILES string and a summary of structural changes.

``reorder_charged_species(df, smiles_col)``
   Reorders charged components in the selected DataFrame column.

``load_models_and_metadata(model_dir)``
   Loads ``model_1`` through ``model_5`` and their corresponding JSON metadata.

``calculate_descriptors(smiles, required_descriptors)``
   Calculates the requested Mordred descriptors and returns their mean vector.

``process_il_smiles_list(smiles_list, required_descriptors)``
   Builds a descriptor matrix for all input molecules.

``main(args)``
   Coordinates input validation, preprocessing, prediction, ensemble statistics,
   and CSV output.

The pure ionic-liquid scripts omit mole fraction from their feature matrix.
The mixture scripts append both temperature and mole fraction. The shared
``models/prediction_settings.py`` module provides ``configure_prediction_args``
and ``_prompt`` for interactive startup when ``--input_csv`` is omitted.

Refractive index models
-----------------------

Ethanol
~~~~~~~

.. literalinclude:: ../models/ri_ethanol/apply_ri_ethanol.py
   :language: python
   :linenos:

Isopropanol
~~~~~~~~~~~

.. literalinclude:: ../models/ri_isopropanol/apply_ri_isopropanol.py
   :language: python
   :linenos:

Water
~~~~~

.. literalinclude:: ../models/ri_water/apply_ri_water.py
   :language: python
   :linenos:

Pure ionic liquid
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../models/ri_pure/apply_ri_pure.py
   :language: python
   :linenos:

Density models
--------------

Ethanol
~~~~~~~

.. literalinclude:: ../models/density_ethanol/apply_density_ethanol.py
   :language: python
   :linenos:

Isopropanol
~~~~~~~~~~~

.. literalinclude:: ../models/density_isopropanol/apply_density_isopropanol.py
   :language: python
   :linenos:

Water
~~~~~

.. literalinclude:: ../models/density_water/apply_density_water.py
   :language: python
   :linenos:

Pure ionic liquid
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../models/density_pure/apply_density_pure.py
   :language: python
   :linenos: