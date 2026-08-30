Streamlit App
=============

:mod:`qspr_il.app` is a Streamlit GUI over the same prediction engine used by
the CLI (:mod:`qspr_il.models.engine`). It supports two input modes: uploading
a CSV of many rows, or entering a single ionic-liquid SMILES.

Running locally
----------------

Install the ``gui`` extra and start the app::

   python -m pip install -e ".[gui]"
   streamlit run qspr_il/app.py

Deploying to Hugging Face Spaces
----------------------------------

The app is deployed to `Hugging Face Spaces <https://huggingface.co/spaces>`_
rather than GitHub, using the standalone export in ``huggingface_space/`` at
the repository root:

.. code-block:: text

   huggingface_space/
     README.md          # Spaces config frontmatter (sdk: streamlit, app_file: app.py, ...)
     app.py              # thin shim: from qspr_il.app import main; main()
     requirements.txt    # streamlit + qspr_il installed from this GitHub repo

A separate folder is used (rather than syncing the whole repository as a
Space) because Hugging Face Spaces reads its configuration from a
``README.md`` at the Space's own root -- reusing the project's real
``README.md`` for that would overwrite its purpose as project documentation.

To deploy:

1. Create a new Space on Hugging Face with the Streamlit SDK.
2. Push the contents of ``huggingface_space/`` to the Space's git remote, e.g.::

      git clone https://huggingface.co/spaces/<user>/<space-name> hf-space
      cp -r huggingface_space/* hf-space/
      cd hf-space
      git add . && git commit -m "Deploy QSPR Streamlit app" && git push

The Space's ``requirements.txt`` installs ``qspr_il`` directly from this
GitHub repository (including its bundled trained model artifacts, shipped as
package data), so the trained models are never duplicated into the Space
export itself.
