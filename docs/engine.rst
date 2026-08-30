Prediction Engine
==================

All 8 prediction models (2 properties x 4 solvent systems) share a single
implementation: :mod:`qspr_il.models.engine`, parametrized by a
:class:`~qspr_il.registry.ModelSpec` from :mod:`qspr_il.registry`. This
replaces what used to be 8 near-identical, independently-maintained scripts.

Registry
--------

.. automodule:: qspr_il.registry
   :members:
   :undoc-members:
   :show-inheritance:

Engine
------

.. automodule:: qspr_il.models.engine
   :members:
   :undoc-members:
   :show-inheritance:

Command-line interface
-----------------------

.. automodule:: qspr_il.cli
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

From the project root::

   python qspr.py

...or, once the package is installed (``pip install -e .``), the console
script::

   qspr-il-predict --model 5 --input_csv datasets/external_test_set.csv \
       --smiles_col IL_SMILES --mole_fraction_col Mole_fraction_IL

Both are equivalent thin wrappers around :func:`qspr_il.cli.main`, which in
turn calls :func:`qspr_il.models.engine.run_prediction` in-process -- there is
no subprocess dispatch and no ``sys.path`` manipulation involved.
