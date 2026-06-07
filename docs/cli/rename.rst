.. _rename_command:

Rename Command
==============

Use this command to rename a dataset. A renamed dataset is a distinct
artefact from its parent, so the command:

1. Assigns a **new** ``uuid`` to the dataset metadata.
2. Updates the dataset **name** stored in the recipe metadata to match the
   new path (the basename of the target without the ``.zarr`` extension).
3. **Moves** the store from the source path to the target path.

Both the source and target paths must end in ``.zarr``. This command operates
on local stores; use the :ref:`Copy Command <copy_command>` for remote stores.

It works for both gridded and tabular datasets, as it only updates the store
attributes and moves the directory.

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: rename
