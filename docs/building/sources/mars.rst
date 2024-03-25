The ``mars`` source will retrieve the data from the ECMWF MARS archive.
For that, you need to have an ECMWF account and build your dataset on
one of the Centre's computers, or use the ``ecmwfapi`` Python package.

The `yaml` block can contain any keys that following the `MARS language
specification`_, with the exception of the ``date``, ``time``` and
``step``.

The missing keys will be filled with the default values, as defined in
the MARS language specification.

.. code:: yaml

   mars:
       levtype: sfc
       param: [2t, msl]
       grid: [0.25, 0.25]

Data from several levels types must be requested in separate requests,
with the ``join`` command.

.. code:: yaml

   join:

    - mars:
        levtype: sfc
        param: [2t, msl]
        grid: [0.25, 0.25]

    - mars:
        levtype: pl
        param: [u, v]
        grid: [0.25, 0.25]

See :ref:`naming-variables` for information on how to name the variables
when mixing single level and multi-levels variables in the same dataset.

.. _mars language specification: https://confluence.ecmwf.int/display/UDOC/MARS+user+documentation
