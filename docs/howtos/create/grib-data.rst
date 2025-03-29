.. _create-grib-data:

#################################
 Create a dataset from GRIB data
#################################

To create a dataset from GRIB files, use the :ref:`grib <grib_source>`
source.

.. literalinclude:: yaml/grib-recipe1.yaml

This recipe will create a dataset with all the GRIB messages present in
the file, whose *valid date* match the requested dates. This means that
for forecast data, the date at which the data are valid is usually the
reference date of the forecast (starting date) plus the forecast step.
