.. _opening-datasets:

##################
 Opening datasets
##################

*******************
 Defining datasets
*******************

The simplest way to open a dataset is to use the `open_dataset`
function:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(dataset, option1=value1, option2=value2, ...)

In that example, `dataset` can be:

-  a local path to a dataset on disk: ``"/path/to/dataset.zarr"``,

-  a URL to a dataset in the cloud: ``"https://path/to/dataset.zarr"``
   or ``"s3://path/to/dataset.zarr"``

-  a dataset name ``"dataset_name"``, which is a string that identifies
   a dataset in the `anemoi` :ref:`configuration file <configuration>`.

-  an already opened dataset. In that case, the function use the options
   to return a modified the dataset, for example with a different time
   range or frequency.

-  a dictionary with a ``dataset`` key that can be any of the above, and
   the remaining keys being the options.

-  a list of any of the above that will be combined either by
   concatenation or joining, based on their compatibility.

-  a combining keyword, such as `join`, `concat`, `ensembles`, etc.
   followed by a list of the above. See :ref:`combining-datasets` for
   more information.

*********
 Options
*********

The general syntax is:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(dataset, option1=value1, option2=value2, ...)

or

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(combine=[dataset1,
                              dataset2, ...])

or

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(combine=[dataset1,
                              dataset2, ...],
                              option1=value1, option2=value2, ...)

The term `combine` is one of `join`, `concat`, `ensembles`, etc. See
:ref:`combining-datasets` for more information.

.. note::

   The options `option1`, `option2`, apply to the combined dataset.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(combine=[{"dataset": dataset1,
                               "option1"=value1, "option2"=value2, ...},
                              {"dataset": dataset2,
                               "option3"=value3, "option4"=value4, ...},
                              ...])

.. note::

   The options `option1`, `option2`, apply to the first dataset, and
   `option3`, `option4`, to the second dataset, etc.
