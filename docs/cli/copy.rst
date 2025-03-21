.. _copy_command:

Copy Command
============


Copying a dataset from one location to another can be error-prone and
time-consuming. This command-line script allows for incremental copying.
When the copying process fails, it can be resumed. It can be used to copy
files from a local directory to a remote server, from a remote server to a
local directory as long as there is a zarr backend to read and write the data.

The script uses multiple threads to make the process faster. However, it is
important to consider that making parallel requests to the same server may
not be ideal, for instance if the server internally uses a limited number of
threads to handle requests.

The option to rechunk the data is available, which can be useful when the
data is stored on a platform that does not support having many small files
or many files in the same directory. However keep in mind that rechunking
has a huge impact on the performance when reading the data: The chunk pattern
for the source dataset has been defined for good reasons, and changing it is
very likely to have a negative impact on the performance.

.. warning::

    When resuming the copying process (using ``--resume``), calling the script with the same arguments for ``--block-size`` and ``--rechunk`` is recommended.
    Using different values for these arguments to resume copying the same dataset may lead to unexpected behavior.


.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: copy
