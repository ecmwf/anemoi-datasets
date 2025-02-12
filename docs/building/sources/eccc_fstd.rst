###########
 eccc-fstd
###########

To read files in ECCC standard format, use the following source:

.. literalinclude:: yaml/eccc_fstd.yaml
   :language: yaml

The recipe will build a dataset from a standard file using the
``fstd2nc`` xarray plugin.

The ``fstd2nc`` dependencie is not part of the default anemoi-datasets
installation and has to be install following the `fstd2nc project
description <https://github.com/neishm/fstd2nc>`_.
