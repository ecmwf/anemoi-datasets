######
 mars
######

The ``mars`` source will retrieve the data from the ECMWF MARS archive.
For that, you need to have an ECMWF account and build your dataset on
one of the Centre's computers, or use the ``ecmwfapi`` Python package.

The `yaml` block can contain any keys that follow the `MARS language
specification`_, with the exception of the ``date``, ``time``, and
``step``.

The missing keys will be filled with the default values, as defined in
the MARS language specification.

.. literalinclude:: yaml/mars1.yaml
   :language: yaml

Data from several level types must be requested in separate requests,
with the ``join`` command.

.. literalinclude:: yaml/mars2.yaml
   :language: yaml

See :ref:`naming-variables` for information on how to name the variables
when mixing single-level and multi-level variables in the same dataset.

.. _mars language specification: https://confluence.ecmwf.int/display/UDOC/MARS+user+documentation
