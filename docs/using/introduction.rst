.. _using-introduction:

###########################
 Using an existing dataset
###########################

An *Anemoi* dataset is a thin wrapper around a zarr_ store that is
optimised for training data-driven weather forecasting models. It is
organised in such a way that I/O operations are minimised (see
:ref:`overview`).

.. _zarr: https://zarr.readthedocs.io/

To open a dataset, you can use the `open_dataset` function.

.. literalinclude:: code/intro_open.py

You can then access the data in the dataset using the `ds` object as if
it were a NumPy array.

.. literalinclude:: code/intro_access_.py

One of the main features of the *anemoi-datasets* package is the ability
to subset or combine datasets.

.. literalinclude:: code/intro_subset.py

In that case, a dataset is created that only contains the data between
the years 2000 and 2020. Combining is done by passing multiple paths to
the `open_dataset` function:

.. literalinclude:: code/intro_combine.py

In the latter case, the datasets are combined along the time dimension
or the variable dimension depending on the dataset's structure.

These operations *compose*: the result of one is itself a dataset that
can be passed to another. See :ref:`using-how-it-works` for an
explanation of how ``open_dataset`` turns its arguments into a network
of interacting objects.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using datasets

   opening
   how-it-works
   synthetic
   methods
   subsetting
   combining
   selecting
   ensembles
   grids
   zip
   statistics
   missing
   other
   matching
   miscellaneous
   configuration
   window
   parameters
