.. _anemoi-dataset_source:

################
 anemoi-dataset
################

.. admonition:: Experimental
   :class: important

      This source is experimental and may change in the future.

An anemoi-dataset can be a source for a dataset:

.. literalinclude:: yaml/anemoi-dataset.yaml
   :language: yaml

The parameters are the same as those used in the ``open_dataset``
function, which allows you to subset and combine datasets. See
:ref:`opening-datasets` for more information.

In particular, this is how local zarr datasets created with anemoi in a
can be used as a source, contrary to :ref:`xarray-zarr` :

.. literalinclude:: yaml/anemoi-zarr-dataset.yaml
   :language: yaml
