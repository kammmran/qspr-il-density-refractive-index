QSPR Modeling of Density and Refractive Index
==============================================

This project provides ensemble QSPR models for predicting ionic-liquid density
and refractive index in water, ethanol, isopropanol, and pure ionic liquids.

Project background
------------------

Ionic liquids are tunable organic salts with negligible vapor pressure, high
thermal stability, and strong solvating ability. Their large chemical space
makes accurate property estimation valuable for designing IL-solvent systems.
This project provides data-driven QSPR estimates for density and refractive
index across composition and temperature ranges.

The data sources, curation procedure, descriptor construction, and model
development workflow are described in the Datasets and Models section below.

Authors
-------

* `Shamkhal Baybekov <https://github.com/sbaybekov>`_ | `LinkedIn <https://www.linkedin.com/in/shamkhal-baybekov/>`_
* `Kamran Heydarov <https://github.com/kammmran>`_ | `LinkedIn <https://www.linkedin.com/in/kamranheydarov/>`_

Quick start
-----------

Create a virtual environment and install the project dependencies::

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

Start the interactive launcher from the project root::

   python qspr.py

The launcher lets you choose the property and solvent, enter CSV column names,
and select an output path. Press Enter to use the displayed default. Generated
predictions are saved under ``results/`` by default.

Input columns
-------------

The bundled external test set uses ``IL_SMILES`` and ``Mole_fraction_IL``.
Temperature is optional; when it is not supplied, the models use 298.15 K.
Pure ionic-liquid models do not require a mole-fraction column.

Repository structure
--------------------

* ``datasets/`` contains curated training sets and the external test set.
* ``models/`` contains application scripts, metadata, and trained ensembles.
* ``figures/`` contains project figures.
* ``results/`` contains generated prediction files and analysis artifacts.

Datasets and Models
-------------------

Training datasets were extracted from the `ILThermo database
<https://ilthermo.boulder.nist.gov/>`_ using the specifically developed
`pyIonics tool <https://github.com/kammmran/pyionics>`_.

The curation procedure included duplicate removal, consistency checks, and
standardization of molecular structures and composition variables.

Molecular structures were represented using 2D descriptors calculated with the
`Mordred descriptor package <https://github.com/mordred-descriptor/mordred>`_.
For each ionic liquid, descriptors were computed separately for the cation and
anion and then averaged to obtain a unified IL representation. Low-variance and
near-constant descriptors were removed, highly correlated features were grouped,
and thermodynamic variables, including temperature and IL mole fraction, were
appended to form the final feature set.

Model development was based on five independently shuffled variants of each
curated dataset. For each variant, 5-fold cross-validation with group-based
splitting was used for XGBoost hyperparameter optimization. The group split
ensured that identical IL SMILES did not appear in both training and validation
folds. The best configuration from each run was retained, producing an ensemble
of five independently trained XGBoost models. Their predictions are combined
into a consensus mean and standard deviation.

Model performance
-----------------

The reported cross-validation performance is summarized below. Density values
are reported in kg/m3 and refractive-index values are for the Na D-line.

.. list-table::
   :widths: 2 2 1 1
   :header-rows: 1

   * - Property
     - System
     - 5-fold RMSE
     - 5-fold R2
   * - Density
     - Pure IL
     - 33.19
     - 0.96
   * - Density
     - IL-water
     - 32.42
     - 0.92
   * - Density
     - IL-ethanol
     - 51.73
     - 0.90
   * - Density
     - IL-isopropanol
     - 38.03
     - 0.94
   * - Refractive index
     - Pure IL
     - 0.01
     - 0.93
   * - Refractive index
     - IL-water
     - 0.01
     - 0.93
   * - Refractive index
     - IL-ethanol
     - 0.01
     - 0.93
   * - Refractive index
     - IL-isopropanol
     - 0.01
     - 0.90

Installation
------------

The project requires Python 3.11 or newer. From the repository root::

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Usage
-----

Start the interactive launcher::

    python qspr.py

Choose a property and solvent, then enter the input CSV and column settings.
Press Enter to keep a displayed default. Predictions are saved under
``results/`` with a model-specific name such as
``results/ri_ethanol_prediction.csv``.

For a direct model script, use command-line options::

    python models/ri_ethanol/apply_ri_ethanol.py --input_csv datasets/external_test_set.csv --smiles_col IL_SMILES --mole_fraction_col Mole_fraction_IL --output_csv results/ri_ethanol_prediction.csv

Temperature is optional and defaults to 298.15 K when the column is absent or
contains missing values. Pure ionic-liquid models do not require a mole-fraction
column.

Further reading
---------------

See the project README for model performance, dataset details, command-line
examples, and the complete repository structure.

Model source files
------------------

* :doc:`functions` - detailed function explanations and source excerpts
* :doc:`data` - dataset sizes, ranges, and missing-value statistics
* :doc:`help_files` - command-line help and example commands
* :doc:`models` - model application source files

.. toctree::
  :hidden:
  :maxdepth: 1

  functions
  data
  help_files
  models