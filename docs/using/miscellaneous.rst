.. _miscellaneous:

#########################
 Miscellaneous functions
#########################

The two functions below can be used to temporarily modify the
:ref:`configuration <configuration>` so that the packages can find named
datasets at given locations.

Use ``add_dataset_path`` to add a path to the list of paths where the
package searches for datasets:

.. _add_dataset_path:

.. code:: python

   from anemoi.datasets import add_dataset_path
   from anemoi.datasets import open_dataset

   add_dataset_path("https://object-store.os-api.cci1.ecmwf.int/ml-examples/")

   ds = open_dataset("an-oper-2023-2023-2p5-6h-v1")

Use ``add_named_dataset`` to add a named dataset to the list of named
datasets:

.. _add_named_dataset:

.. code:: python

   from anemoi.datasets import add_named_dataset
   from anemoi.datasets import open_dataset

   add_named_dataset(
       "example-dataset",
       "https://object-store.os-api.cci1.ecmwf.int/ml-examples/an-oper-2023-2023-2p5-6h-v1.zarr",
   )

   ds = open_dataset("example-dataset")
