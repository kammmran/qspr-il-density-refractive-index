Help Files and Example Commands
===============================

Each model directory contains two small text files:

``code_help_message.txt``
   A saved copy of the script's ``--help`` output. It lists the available
   command-line flags, their meanings, and the model-specific defaults.

``example_cmd.txt``
   A ready-to-adapt command showing how to run that model with the bundled
   external test-set column names.

The Python scripts are the source of truth. If a command-line option changes,
regenerate the corresponding help snapshot rather than relying on an older
text file. The root launcher, ``python qspr.py``, is the easiest interactive
alternative.

Command-line options
--------------------

``--input_csv``
   Path to a comma-separated input file. It can be omitted when using the
   interactive prompt; the prompt then asks for it.

``--smiles_col``
   Name of the column containing ionic-liquid SMILES. The standalone scripts
   default to ``SMILES``. The bundled external test set uses ``IL_SMILES``.

``--mole_fraction_col``
   Name of the IL mole-fraction column for mixture models. The standalone
   scripts default to ``Mole_fraction``; the bundled test set uses
   ``Mole_fraction_IL``. Pure ionic-liquid models do not use this option.

``--temp_col``
   Name of the temperature column. Missing or omitted temperature values use
   298.15 K.

``--model_dir``
   Directory containing five ``model_*.joblib`` files and five matching
   ``metadata_*.json`` files.

``--output_csv``
   Destination CSV for the original input columns plus standardized SMILES,
   ``prediction_mean``, and ``prediction_std``.

Running a saved example
------------------------

Run commands from the model directory, or change the paths when running from
the project root. For example::

   cd models/density_ethanol
   python apply_density_ethanol.py --input_csv ../../datasets/external_test_set.csv --smiles_col IL_SMILES --mole_fraction_col Mole_fraction_IL --model_dir density_ethanol_ensemble_model --output_csv ../../results/density_ethanol_prediction.csv

The interactive equivalent is::

   python ../../qspr.py

Refractive index examples
-------------------------

Ethanol
~~~~~~~

.. literalinclude:: ../models/ri_ethanol/code_help_message.txt
   :language: text

.. literalinclude:: ../models/ri_ethanol/example_cmd.txt
   :language: console

Isopropanol
~~~~~~~~~~~

.. literalinclude:: ../models/ri_isopropanol/code_help_message.txt
   :language: text

.. literalinclude:: ../models/ri_isopropanol/example_cmd.txt
   :language: console

Water
~~~~~

.. literalinclude:: ../models/ri_water/code_help_message.txt
   :language: text

.. literalinclude:: ../models/ri_water/example_cmd.txt
   :language: console

Pure ionic liquid
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../models/ri_pure/code_help_message.txt
   :language: text

.. literalinclude:: ../models/ri_pure/example_cmd.txt
   :language: console

Density examples
----------------

Ethanol
~~~~~~~

.. literalinclude:: ../models/density_ethanol/code_help_message.txt
   :language: text

.. literalinclude:: ../models/density_ethanol/example_cmd.txt
   :language: console

Isopropanol
~~~~~~~~~~~

.. literalinclude:: ../models/density_isopropanol/code_help_message.txt
   :language: text

.. literalinclude:: ../models/density_isopropanol/example_cmd.txt
   :language: console

Water
~~~~~

.. literalinclude:: ../models/density_water/code_help_message.txt
   :language: text

.. literalinclude:: ../models/density_water/example_cmd.txt
   :language: console

Pure ionic liquid
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../models/density_pure/code_help_message.txt
   :language: text

.. literalinclude:: ../models/density_pure/example_cmd.txt
   :language: console