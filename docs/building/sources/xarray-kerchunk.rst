#################
 xarray-kerchunk
#################

This source is for building datasets from virtualised data in the form of
`Kerchunk`_ references.

Input references are supported as JSON strings, local and remote JSON file
paths, and dictionaries.
For remote file paths requiring configuration (e.g. authentication) to
open, use the ``json_options`` field to pass relevant options to
`fsspec.open`_.
For loading remote data from references, use the ``storage_options`` field
to pass relevant options to `xarray.open_dataset`_'s ``backend_kwargs``
parameter.

The below example recipe demonstrates building a dataset from a Kerchunk
references JSON file stored in a private Azure Blob Storage container.
Here, the JSON file and internally referenced data are stored in the same
container requiring the same storage options.
This recipe makes use of YAML anchors to avoid repetition of the same
options in both fields.

.. literalinclude:: yaml/xarray-kerchunk.yaml
   :language: yaml

.. warning::
   Secrets included in the recipe are stored in plain text in the output
   dataset's metadata (.zattrs file).
   Prefer short-lived credentials (e.g. SAS tokens) and remove secrets
   yourself where necessary.

The code below is inspired by the `Kerchunk tutorial`_, and makes use of
a subset of the `ERA5 dataset available on AWS`_. You may need to
install the relevant packages before running the code below.

.. literalinclude:: xarray-kerchunk.py
   :language: python

See :ref:`create-cf-data` for more information.

.. _kerchunk: https://fsspec.github.io/kerchunk/

.. _fsspec.open: https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html

.. _era5 dataset available on aws: https://registry.opendata.aws/ecmwf-era5/

.. _kerchunk tutorial: https://fsspec.github.io/kerchunk/tutorial.html
