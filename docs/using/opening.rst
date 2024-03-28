.. _opening-datasets:

##################
 Opening datasets
##################

The simplest way to open a dataset is to use the `open_dataset`
function:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(dataset, option1=value1, option2=value2, ...)

In that example, `dataset` can be:

-  a local path to a dataset on disk:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("/path/to/dataset.zarr")

-  a URL to a dataset in the cloud:

.. code:: python

   from anemoi.datasets import open_dataset

   ds1 = open_dataset("https://path/to/dataset.zarr")

   ds2 = open_dataset("s3://path/to/dataset.zarr")

-  a dataset name, which is a string that identifies a dataset in the
   `anemoi` :ref:`configuration file <configuration>`.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("dataset_name")

-  an already opened dataset. In that case, the function use the options
   to return a modified the dataset, for example with a different time
   range or frequency.

.. code:: python

   from anemoi.datasets import open_dataset

   ds1 = open_dataset("/path/to/dataset.zarr")

   ds2 = open_dataset(ds1, frequency="24h", begin="2000", end="2010")

-  a dictionary with a ``dataset`` key that can be any of the above, and
   the remaining keys being the options. The purpose of this option is
   to allow the user to open a dataset based on a configuration file.
   See `an example <open_with_config>`_ below

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset({"dataset": dataset,
                      "option1": value1,
                      "option2": value2, ...})

-  a list of any of the above that will be combined either by
   concatenation or joining, based on their compatibility.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset([dataset1, dataset2, ...])

-  a combining keyword, such as `join`, `concat`, `ensembles`, etc.
   followed by a list of the above. See :ref:`combining-datasets` for
   more information.

.. literalinclude:: open_combine1_.py
   :language: python

.. note::

   In the example above, the options `option1`, `option2`, apply to the
   combined dataset. To apply options to individual datasets, use a list
   of dictionaries as shown below. The options `option1`, `option2`,
   apply to the first dataset, and `option3`, `option4`, to the second
   dataset, etc.

.. literalinclude:: open_combine2_.py
   :language: python

.. _open_with_config:

As mentioned above you, using the dictionary to open a dataset can be
useful for software that provide users with the ability to define their
requirements in a configuration file:

.. code:: python

   with open("config.yaml") as file:
       config = yaml.safe_load(file)

   ds = open_dataset(config)

The dictionary can be a complex as needed, for example:

.. literalinclude:: open_complex.py
   :language: python

..
   TODO:
   When opening a complex dataset the user can use the `adjust` keyword to
   let the function know how to combine the datasets. The `combine` keyword
   can be any of the following:
