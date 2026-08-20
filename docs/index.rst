QSPR Modeling of Density and Refractive Index
==============================================

This project provides ensemble QSPR models for predicting ionic-liquid density
and refractive index in water, ethanol, isopropanol, and pure ionic liquids.

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

Further reading
---------------

See the project README for model performance, dataset details, command-line
examples, and the complete repository structure.