#########
 Options
#########

These are equivalent:

.. code:: python

   ds = open_dataset(path)
   ds = open_dataset(dataset=path)
   ds = open_dataset({"dataset": path})

The last example is useful when the dataset is defined from a
configuration file:

.. code:: python

   with open("config.yaml") as file:
       config = yaml.safe_load(file)

   ds = open_dataset(config)

When defining a dataset from another, you can either use a path or a
dataset:

.. code:: python

   open_dataset(path, statistics=other_path)
   open_dataset(path, statistics=other_dataset)
   open_dataset(path, statistics={"dataset": other_path, "...": ...})

This also applies when combining datasets:

.. code:: python

   open_dataset(ensembles=[dataset1, dataset2, ...])
   open_dataset(ensembles=[path1, path2, ...])
   open_dataset(ensembles=[dataset1, path2, ...])
   open_dataset(
       ensembles=[
           {"dataset": path1, "...": ...},
           {"dataset": path2, "...": ...},
           ...,
       ]
   )

*********
 Options
*********

.. code:: python

   open_dataset(
       dataset,
       start=None,
       end=None,
       frequency=None,
       select=None,
       drop=None,
       reorder=None,
       rename=None,
       statistics=None,
       thinning=None,
       area=None,
       ensembles=None,
       grids=None,
       method=None,
   )

dataset
=======

This is a path or URL to a ``zarr`` file that has been created with this
package, as described in :ref:`Building training datasets
<building-introduction>`.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2")
   ds = open_dataset(
       "/path/to/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2.zarr"
   )
   ds = open_dataset(
       "https://example.com/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2.zarr"
   )
   ds = open_dataset("s3://bucket/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2.zarr")

Alternatively, you can pass an already opened dataset:

.. code:: python

   from anemoi.datasets import open_dataset

   ds1 = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2")
   ds2 = open_dataset(ds1, start=1979, end=2020)
