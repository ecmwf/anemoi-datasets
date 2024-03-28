.. _configuration:

###############
 Configuration
###############

..
   The configuration file is a YAML

..
   file that contains a list of datasets. Each dataset is a dictionary with

..
   a `name` key and a `path` key. The `name` key is a string that

..
   identifies the dataset, and the `path` key is a string that contains the

..
   path to the dataset. The `open_dataset` function looks for the dataset

..
   name in the configuration file and opens the dataset with the

..
   corresponding path.

.. code:: toml

   [datasets]
   dataset1 = "/path/to/dataset1.zarr"
   dataset2 = "/path/to/dataset2.zarr"
   "*" = "/path/to/{name}.zarr"

   [dataset.path]
   - name = "dataset1"
     path = "/path/to/dataset1.zarr"
