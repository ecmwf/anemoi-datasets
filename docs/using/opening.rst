.. _opening-datasets:

##################
 Opening datasets
##################

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset("path/to/dataset.zarr", option1=value1, option2=value2, ...)

or

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset(combine=["path/to/dataset1.zarr",
                              "path/to/dataset2.zarr", ...])

or

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset(combine=["path/to/dataset1.zarr",
                              "path/to/dataset2.zarr", ...],
                              option1=value1, option2=value2, ...)

The term `combine` is one of `join`, `concat`, `ensembles`, etc. See
:ref:`combining-datasets` for more information.

.. note::

   The options `option1`, `option2`, apply to the combined dataset.

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset(combine=[{"dataset": "path/to/dataset1.zarr",
                               "option1"=value1, "option2"=value2, ...},
                              {"dataset": "path/to/dataset2.zarr",
                               "option3"=value3, "option4"=value4, ...},
                              ...])

.. note::

   The options `option1`, `option2`, apply to the first dataset, and
   `option3`, `option4`, to the second dataset, etc.
