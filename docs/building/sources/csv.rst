#####
 csv
#####

This source reads data from a CSV file using ``Pandas``. The ``flavour``
parameter allows users to specify which columns represent the date,
latitude, and longitude, which are used by anemoi to create the dataset.

The ``columns`` parameter allows users to specify which columns to read
from the CSV file and how to rename them. The columns provided in the
``flavour`` parameter should not be included in the ``columns``
parameter.

The default flavour is ``date``, ``latitude``, and ``longitude``, which
means that the source will look for columns with those names in the CSV
file and assumes that the ``date`` is a full ``datetime`` column.

In the example below, the date is represented in the CSV file as two
columns, named ``date`` and ``time``, so we specify them as a list.

.. literalinclude:: yaml/csv.yaml
   :language: yaml
