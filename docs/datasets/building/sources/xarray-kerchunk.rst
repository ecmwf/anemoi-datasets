#################
 xarray-kerchunk
#################

.. literalinclude:: yaml/xarray-kerchunk.yaml
   :language: yaml

The code below is inspired by the `kerchunk tutorial`_, and makes use of
a subset of the `ERA5 dataset available on AWS`_. You may need to
install the relevant packages before running the code below.

.. literalinclude:: xarray-kerchunk.py
   :language: python

See :ref:`create-cf-data` for more information.

.. _era5 dataset available on aws: https://registry.opendata.aws/ecmwf-era5/

.. _kerchunk tutorial: https://fsspec.github.io/kerchunk/tutorial.html
