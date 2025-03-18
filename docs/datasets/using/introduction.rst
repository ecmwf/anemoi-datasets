.. _using-introduction:

##############
 Introduction
##############

An *Anemoi* dataset is a thin wrapper around a zarr_ store that is
optimised for training data-driven weather forecasting models. It is
organised in such a way that I/O operations are minimised (see
:ref:`overview`).

.. _zarr: https://zarr.readthedocs.io/

To open a dataset, you can use the `open_dataset` function.

.. literalinclude:: code/open_path.py

You can then access the data in the dataset using the `ds` object as if
it was a NumPy array.

.. literalinclude:: code/some_attributes_.py

One of the main features of the *anemoi-datasets* package is the ability
to subset or combine datasets.

.. literalinclude:: code/subset_example.py

In that case, a dataset is created that only contains the data between
the years 2000 and 2020. Combining is done by passing multiple paths to
the `open_dataset` function:

.. literalinclude:: code/combine_example.py
   :language: python

In the latter case, the datasets are combined along the time dimension
or the variable dimension depending on the dataset's structure.
