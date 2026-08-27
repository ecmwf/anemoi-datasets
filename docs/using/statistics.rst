.. _using-statistics:

############
 Statistics
############

When combining datasets, the statistics are not recomputed. Instead, the
statistics of the first dataset encountered are returned by the
``statistics`` property.

You can change that behaviour by using the `statistics` option to select
a specific dataset from which to get the statistics:

.. literalinclude:: code/statistics_select_.py

*********************
 Residual statistics
*********************

.. warning::

   Experimental: the ``residual_statistics`` option and the
   ``residual_statistics`` attribute may be removed or renamed in a
   future release.

*Residual statistics* are the statistics of the difference between two
datasets, e.g. between a high-resolution dataset regridded to a coarser
grid and the native dataset on that grid. They do not live in a Zarr
store: they are computed once with the :ref:`compute command
<compute_command>` and written to a JSON file:

.. code-block:: bash

    anemoi-datasets compute hi-res grid=o96 \
        --statistics-residual lo-res --output residual.json

That file is then attached to a dataset with the ``residual_statistics``
option, and read back from the ``residual_statistics`` property:

.. literalinclude:: code/statistics_residual_.py

The file records the two datasets it was computed from, and is marked as
holding residuals, so a plain statistics file is rejected rather than
silently used. It must cover every variable of the dataset it is attached
to; extra variables in the file are ignored.

Like ``statistics``, residual statistics are indexed by variable only, so
they follow the usual operations: selecting, dropping or reordering
variables re-indexes them, a ``join`` concatenates them, and ``rescale``
scales them (a residual is a difference, so the offset cancels out and
only the scale applies). Datasets with no residual statistics attached
raise ``ResidualStatisticsNotAvailable`` when the property is accessed.
