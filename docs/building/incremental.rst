.. _create-incremental:

################################
 Create a dataset incrementally
################################

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

.. code:: bash

   anemoi-datasets load dataset.zarr --part 1/20

.. code:: bash

   anemoi-datasets load dataset.zarr --part 2/20

.. code:: bash

   anemoi-datasets finalise dataset.zarr

************
 Additions:
************

anemoi-datasets init-additions dataset.zarr --delta 12h anemoi-datasets
load-additions dataset.zarr --part 1/2 --delta 12h anemoi-datasets
load-additions dataset.zarr --part 2/2 --delta 12h

anemoi-datasets init-additions dataset.zarr --delta 6h anemoi-datasets
load-additions dataset.zarr --part 1/2 --delta 6h anemoi-datasets
load-additions dataset.zarr --part 2/2 --delta 6h

anemoi-datasets finalise-additions dataset.zarr --delta 6h 12h

********
 Patch:
********

anemoi-datasets patch $NAME.zarr

**********
 Cleanup:
**********

anemoi-datasets cleanup $NAME.zarr # delete temp files
