.. _opening-datasets:

##################
 Opening datasets
##################

The simplest way to open a dataset is to use the `open_dataset`
function:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(dataset, option1=value1, option2=...)

In this example, `dataset` can be:

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

-  an already opened dataset. In that case, the function uses the
   options to return a modified dataset, for example with a different
   time range or frequency.

.. code:: python

   from anemoi.datasets import open_dataset

   ds1 = open_dataset("/path/to/dataset.zarr")

   ds2 = open_dataset(ds1, frequency="24h", start="2000", end="2010")

-  a dictionary with a ``dataset`` key that can be any of the above, and
   the remaining keys being the options. The purpose of this option is
   to allow the user to open a dataset based on a configuration file.
   See :ref:`an example <open_with_config>` below:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset({"dataset": dataset, "option1": value1, "option2": ...})

-  a list of any of the above that will be combined either by
   concatenation or joining, based on their compatibility.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset([dataset1, dataset2, ...])

-  a combining keyword, such as `join`, `concat`, `ensembles`, etc.
   followed by a list of the above. See :ref:`combining-datasets` for
   more information.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
      ensemble=[dataset1, dataset2],
      option1=value1,
      option2=...,
   )

.. note::

   In the example above, the options `option1`, `option2`, apply to the
   combined dataset. To apply options to individual datasets, use a list
   of dictionaries as shown below. The options `option1`, `option2`,
   apply to the first dataset, and `option3`, `option4`, to the second
   dataset, etc.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
      combine=[
         {"dataset": dataset1, "option1": value1, "option2": ...},
         {"dataset": dataset2, "option3": value3, "option4": ...},
      ]
   )

.. _open_with_config:

As mentioned above, using the dictionary to open a dataset can be useful
for software that provides users with the ability to define their
requirements in a configuration file:

.. code:: python

   with open("config.yaml") as file:
      config = yaml.safe_load(file)

   ds = open_dataset(config)

The dictionary can be as complex as needed, for example:

.. code:: python

   from anemoi.datasets import open_dataset

   config = {
      "dataset": {
         "ensemble": [
               "/path/to/dataset1.zarr",
               {"dataset": "dataset_name", "end": 2010},
               {"dataset": "s3://path/to/dataset3.zarr", "start": 2000, "end": 2010},
         ],
         "frequency": "24h",
      },
      "select": ["2t", "msl"],
   }

   ds = open_dataset(config)

The `open_dataset` function returns an object that wraps around
`numpy.ndarray`, so it is possible to inspect the dataset and visualise
it with standard Python tools. For example:

.. code:: python

   from anemoi.datasets import open_dataset
   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs

   ds = open_dataset("aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1.zarr", select="2t")
   fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
   p = ax.scatter(x=ds.longitudes, y=ds.latitudes, c=ds[0, 0, 0, :])
   ax.coastlines()
   ax.gridlines(draw_labels=True)
   plt.colorbar(p, label="K", orientation="horizontal")

.. figure:: ../../_static/2t_map_example.png
   :alt: example map plot
   :align: center

..
   TODO:
   When opening a complex dataset the user can use the `adjust` keyword to
   let the function know how to combine the datasets. The `combine` keyword
   can be any of the following:
