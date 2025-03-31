.. _grib-index_command:

Grib-index Command
============

The `grib-index` command is used to create an index file for GRIB files. The index file is then used
by the `grib-index` :ref:`source <grib-index_source>`.

The command will recursively scan the directories provided and open all the GRIB files found. It will
then create an index file for each GRIB file, which will be used to read the data.

.. code:: bash

    anemoi-datasets grib-index --index index.db /path1/to/grib/files /path2/to/grib/files


See :ref:`grib_flavour` for more information about GRIB flavours.


.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: grib-index
