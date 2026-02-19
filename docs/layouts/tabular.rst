.. _layouts-tabular:

#########
 Tabular
#########

Tabular data are typically observations. Each observation has its own
time and location. All observation shoul have the same set of variables,
which can be NaNs for some observations.

***************
 Sample format
***************

Although the underlying Zarr array contains date and position
information, a sample will only return the data values (excluding date,
time, latitude, longitude that are in the Zarr), and will match the
number of variables. This guarantees the same behaviour as for gridded
data.

+--------+------+-----+-------+
| Col 1  | Col  | ... | Col N |
|        | 2    |     |       |
+========+======+=====+=======+
| 1013.2 | 7.5  | ... | 23.5  |
+--------+------+-----+-------+
| 1012.8 | 6.8  | ... | -4.5  |
+--------+------+-----+-------+
| 1014.1 | 5.2  | ... | 12.9  |
+--------+------+-----+-------+
| 1011.7 | 8.0  | ... | 0.0   |
+--------+------+-----+-------+
| 1013.5 | -2.1 | ... | -4.2  |
+--------+------+-----+-------+
| ...    | ...  | ... | ...   |
+--------+------+-----+-------+

It is expected that if the model needs time and space coordinate
information, they are encoded in ``cos_longitude``, ``cos_latitude``,
``cos_julian_day``, ``sin_julian_day``, etc. variables.

.. code:: python

   sample = ds[42]

   # A 2D Array is returned, the first dimension is the number of observations in the 43st window.
   assert len(sample.shape) == 2

   # The second dimension are the variables
   assert sample.shape[1] == len(ds.variables)

   # Same for statistics

   assert len(ds.statistics['mean']) == len(ds.variables)

Auxiliary information can be accessed as:

.. code:: python

   sample = ds[42]

   number_of_observations_in_window = sample.shape[0]


    # Returns the corresponding latitudes

   sample.latitudes

   assert len(sample.latitudes) == number_of_observations_in_window

   # Returns the corresponding longitudes

   sample.longitudes

   assert len(sample.longitudes) == number_of_observations_in_window


   x.dates # Returns the corresponding row dates

   # Returns the corresponding dates

   sample.dates

   assert len(sample.dates) == number_of_observations_in_window

   # Return the reference date of the window

   sample.reference_date

   assert sample.reference_date == ds.start_date + 42 * ds.frequency

   # Return the timedeltas in seconds relative to the reference_date

   sample.timedeltas

   assert len(sample.timedeltas) == number_of_observations_in_window
