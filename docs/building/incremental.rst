.. _create-incremental:

##################################
 Creating a dataset incrementally
##################################

This guide shows how to create a dataset incrementally. This is useful
when you have a large dataset that you want to load in parts, to avoid
running out of memory. Because parts can be loaded in parallel, this can
also speed up the process.

**********************
 Building the dataset
**********************

You first need to create an empty dataset with the `init` command, which
is similar to the `create` command. If there is already a dataset with
the same name, you can use the `--overwrite` flag to replace it. The
`init` command requires a YAML file with the dataset configuration and a
name for the dataset. The content of the YAML file will be copied into
the dataset, so it will not be needed by following commands.

.. code:: bash

   anemoi-datasets init dataset.yaml dataset.zarr --overwrite

You can then load the dataset in parts with the `load` command. You just
pass which part you want to load with the `--part` flag.

.. note::

   Parts are numbered from 1 to N, where N is the total number of parts
   (unlike Python, where they would start at zero). This is to make it
   easier to use the `seq(1)` command in bash.

You can load multiple parts in any order and in parallel by running the
`load` command in different terminals, slurm jobs or any other
parallelisation tool. The library relies on the `zarr` library to handle
concurrent writes.

.. code:: bash

   anemoi-datasets load dataset.zarr --part 1/20

.. code:: bash

   anemoi-datasets load dataset.zarr --part 2/20

... and so on ... until:

.. code:: bash

   anemoi-datasets load dataset.zarr --part 20/20

Once you have loaded all the parts, you can finalise the dataset with
the `finalise` command. This will write the metadata and the attributes
to the dataset, and consolidate the statistics and clean up some
temporary files.

.. code:: bash

   anemoi-datasets finalise dataset.zarr

You can follow the progress of the dataset creation with the `inspect`
command. This will show you the percentage of parts loaded.

.. code:: bash

   anemoi-datasets inspect dataset.zarr

It is possible that some temporary files are left behind at the end of
the process. You can clean them up with the `cleanup` command.

.. code:: bash

   anemoi-datasets cleanup dataset.zarr

***********************
 Additional statistics
***********************

`anemoi-datasets` can compute additional statistics for the dataset,
mostly statistics of the increments between two dates (e.g. 6h or 12h).

To add statistics for 6h increments:

.. code:: bash

   anemoi-datasets init-additions dataset.zarr --delta 6h
   anemoi-datasets load-additions dataset.zarr --part 1/2 --delta 6h
   anemoi-datasets load-additions dataset.zarr --part 2/2 --delta 6h
   anemoi-datasets finalise-additions dataset.zarr --delta 6h

To add statistics for 12h increments:

.. code:: bash

   anemoi-datasets init-additions dataset.zarr --delta 12h
   anemoi-datasets load-additions dataset.zarr --part 1/2 --delta 12h
   anemoi-datasets load-additions dataset.zarr --part 2/2 --delta 12h
   anemoi-datasets finalise-additions dataset.zarr --delta 12h

If this process leaves temporary files behind, you can clean them up
with the `cleanup` command.

.. code:: bash

   anemoi-datasets cleanup dataset.zarr

********************************
 Patching the dataset metadata:
********************************

The following command will patch the dataset metadata. In particular, it
will remove any references to the YAML file used to initialise the
dataset.

.. code:: bash

   anemoi-datasets patch dataset.zarr
