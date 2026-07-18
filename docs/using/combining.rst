.. _combining-datasets:

####################
 Combining datasets
####################

You can create "virtual" datasets by combining two or more datasets. The
combination will behave exactly as if you had a single dataset, with all
the methods behaving as expected. The package will ensure that the data
is loaded lazily from the original datasets.

.. warning::

   When combining datasets, the statistics of the first dataset are used
   by default. You can change this by setting the
   :ref:`using-statistics` option to a different dataset, even if it
   is not part of the combination.

When combining datasets, the package will check that the datasets are
compatible, i.e., that they have the same resolution, the same
variables, etc. The compatibility checks depend on the type of
combination. You can adjust some of the attributes of the datasets to
make them compatible, e.g., by changing their date range or frequency
using :ref:`start`, :ref:`end`, :ref:`frequency`, etc. You can also ask
the package to :ref:`automatically adjust <using-matching>` these
attributes.

***********
 automatic
***********

If you just provide a list of datasets, the package will automatically
attempt to combine them:

-  If the datasets have the same variables, the same ensemble dimension,
   and the same grids, and if the dates, once concatenated, create a
   continuous range of dates with a constant frequency, the package will
   combine using concat_.

-  If the datasets have the same dates, ensemble dimension and grids,
   the package will combine using join_.

.. _concat:

********
 concat
********

You can concatenate two or more datasets along the dates dimension. The
package will check that all datasets are compatible (same resolution,
same variables, etc.). Currently, the datasets must be given in
chronological order with no gaps between them. If you want to
concatenate datasets that have gaps between them, check the
:ref:`fill_missing_gaps <fill_missing_gaps>` option.

.. literalinclude:: code/combining_concat.py

.. image:: ../_static/concat.png
   :align: center
   :alt: Concatenation

Please note that you can pass more than two datasets to the function.

   **NOTE:** When concatenating datasets, the statistics are not
   recomputed; it is the statistics of the first dataset that are
   returned to the user. You can change this using the
   :ref:`using-statistics` option.

When concatenating datasets, the package will check that variables are compatible, i.e.,
that they have the same units, the same time processing (e.g. whether the data is instantaneous or accumulated),
the same time processing period (e.g. whether the data are 3-hourly or 6-hourly accumulations),
the same type of level (e.g. whether the data are on pressure levels or model levels), etc.

You can turn some of the check off:

.. literalinclude:: code/combining_concat_check_.py


You can also turn off checks for individual variables by setting:

.. literalinclude:: code/combining_concat_check_vars_.py


.. _join:

******
 join
******

You can join two datasets that have the same dates, combining their
variables.

.. literalinclude:: code/combining_join.py

.. image:: ../_static/join.png
   :align: center
   :alt: Join

If a variable is present in more than one file, the last occurrence of
that variable will be used and will be at the position of the first
occurrence of that name.

.. image:: ../_static/overlay.png
   :align: center
   :alt: Overlay

Please note that you can join more than two datasets.

.. _ensembles:

***********
 ensembles
***********

You can combine two or more datasets that have the same dates,
variables, grids, etc. along the ensemble dimension. The package will
check that all datasets are compatible.

.. literalinclude:: code/combining_ensembles_.py

.. _grids:

*******
 grids
*******

.. literalinclude:: code/combining_grids_.py

All the grid points are concatenated, in the order they are given. The
`latitudes` and `longitudes` are also concatenated.

********
 cutout
********

.. literalinclude:: code/combining_cutout_.py

The `cutout` combination only supports two datasets. The first dataset
is considered to be a limited area model (LAM), while the second one is
considered to be a global model or boundary conditions. It is therefore
expected that the bounding box of the first dataset is contained within
the bounding box of the second dataset.

The image below shows the global dataset:

.. image:: ../_static/cutout-1.png
   :width: 75%
   :align: center
   :alt: Cutout

The image below shows the LAM dataset:

.. image:: ../_static/cutout-2.png
   :width: 75%
   :align: center
   :alt: Cutout

A 'cutout' is performed by removing the grid points from the global
dataset that are contained in the LAM dataset. The result is shown
below:

.. image:: ../_static/cutout-3.png
   :width: 75%
   :align: center
   :alt: Cutout

The final dataset is the concatenation of the LAM dataset and the
cutout:

.. image:: ../_static/cutout-4.png
   :width: 75%
   :align: center
   :alt: Cutout

You can also pass a `min_distance_km` parameter to the `cutout`
function. Any grid points in the global dataset that are closer than
this distance to a grid point in the LAM dataset will be removed. This
can be useful to control the behaviour of the algorithm at the edge of
the cutout area. If no value is provided, the algorithm will compute its
value as the smallest distance between two grid points in the global
dataset over the cutout area. If you do not want to use this feature,
you can set `min_distance_km=0`, or provide your own value.

The plots below illustrate how the cutout differs if `min_distance_km`
is not given (top) or if `min_distance_km` is set to `0` (bottom). The
difference can be seen at the boundary between the two grids:

.. image:: ../_static/cutout-5.png
   :align: center
   :alt: Cutout

.. image:: ../_static/cutout-6.png
   :align: center
   :alt: Cutout

To debug the combination, you can pass `plot=True` to the `cutout`
function (when running from a Notebook), or use `plot="prefix"` to save
the plots to series of PNG files in the current directory.

.. _complement:

************
 complement
************

That feature will interpolate the variables of `dataset2` that are not
in `dataset1` to the grid of `dataset1`, add them to the list of
variables of `dataset1` and return the result.

.. literalinclude:: code/combining_complement_.py

Currently ``what`` can only be ``variables`` and can be omitted.

The value for ``interpolation`` can be one of ``none`` (default) or
``nearest``. In the case of ``none``, the grids of the two datasets must
match. In case of ``interpolation``, an additional parameter ``k`` can
be set to specify the number of nearest neighbours to use.

This feature was originally designed to be used in conjunction with
``cutout``, where `dataset1` is the lam, and `dataset2` is the global
dataset.

.. literalinclude:: code/combining_complement_combine_.py

Another use case is to simply bring all non-overlapping variables of a
dataset into another:

.. literalinclude:: code/combining_complement_simple_.py
