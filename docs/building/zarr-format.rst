.. _zarr-format:

#############
 Zarr format
#############

Datasets are stored as Zarr_ archives. The Zarr specification exists in
two versions, *Zarr format 2* and *Zarr format 3*. These should not be
confused with the versions of the zarr-python library: zarr-python 3 can
read and write both formats, but zarr-python 2 can only read and write
Zarr format 2.

****************
 Default format
****************

New datasets are written in Zarr format 2, whichever version of
zarr-python is installed. This ensures that they can be read by tools
that still use zarr-python 2, such as current versions of
*anemoi-training* and *anemoi-inference*.

The format of an existing dataset is detected when it is opened, so
datasets in either format can be read, provided that the installed
version of zarr-python supports it.

***********************
 Writing Zarr format 3
***********************

.. warning::

   Writing Zarr format 3 is experimental. Datasets in Zarr format 3
   cannot be read with zarr-python 2, so they cannot be used by tools
   that have not moved to zarr-python 3 yet.

To create a dataset in Zarr format 3, set ``format`` in the
``output`` section of the recipe:

.. code:: yaml

   output:
     format: zarr3

This requires zarr-python 3 or later. With zarr-python 2, the recipe is
rejected when it is loaded, before anything is written.

The format is chosen when the dataset is created by the ``create`` or
``init`` commands. The following steps (``load``, ``finalise``,
``patch``, the ``*-additions`` commands, ...) keep the format of the
existing dataset.

*******************
 Copying a dataset
*******************

When :ref:`copying a dataset <copy_command>` without changing its
chunks, the copy keeps the format of the source. Copies between a remote
location (``ssh://``, ``s3://`` or ``abfs://``) and a local one transfer
the files as they are. Every other copy, including a copy between two
local paths, reads and rewrites the data.

When the data is rewritten (``--rechunk``, ``--reshard`` or
``--obfuscate``), the target is also written in the format of the
source, unless the ``--format`` option selects another one. This
option is only available with zarr-python 3, and can also be used on its
own to convert a dataset:

.. code:: bash

   anemoi-datasets copy --format zarr3 dataset.zarr dataset-v3.zarr

Sharding (``--reshard``) only exists in Zarr format 3, so resharding
writes the target in Zarr format 3 and cannot be combined with
``--format zarr2``.

When resuming an interrupted copy with ``--resume``, the target keeps its
format. If ``--format`` is given and does not match the format of the
target, the copy is aborted. Use ``--overwrite`` to start again instead.

*************
 Compression
*************

The ``data`` array of a new dataset is compressed with Blosc (lz4,
compression level 5, byte shuffle), whatever the Zarr format. Without
an explicit compressor, zarr-python 3 writes the ``data`` array of a
Zarr format 3 dataset with zstd and no shuffle.

When a copy rewrites the data, the ``data`` array of the target keeps the
compressor of the source. Blosc, Zstd and Gzip are translated to the
codecs of the Zarr format of the target, including the compressor of the
chunks inside the shards of a sharded source, and a source without
compression gives a target without compression. If the compressor of the
source has no equivalent in the format of the target (for example LZ4
from a Zarr format 2 source to Zarr format 3), a warning is logged and
the target uses Blosc (lz4, compression level 5, byte shuffle).

**********************************
 Checking the format of a dataset
**********************************

The format of a dataset can be recognised from the metadata file at the
top of the archive: ``.zgroup`` for Zarr format 2, ``zarr.json`` for
Zarr format 3.

.. _zarr: https://zarr.readthedocs.io/
