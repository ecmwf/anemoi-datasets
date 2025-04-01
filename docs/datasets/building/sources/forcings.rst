.. _forcing_variables:

##########
 forcings
##########

The purpose of `forcings` is to provide fields with values that only
depend on the grid cell and/or the time.

Because the source needs to generate fields on the same grids as the
others, it requires a template field. This is provided in the recipe
with the ``template`` keyword:

.. literalinclude:: yaml/forcings.yaml
   :language: yaml

.. _yaml-reference:

The value ``${input.join.0.source1}`` is the "path" to the first source,
starting from the root of the recipe. The path is composed of the nodes
in the recipe, separated by dots. If a node contains a list, the index
of the next node is added after the node name, starting from 0. This is
called a `reference`.

.. warning::

   If you use the :ref:`concat <building-concat>` construct to have
   different inputs for different time ranges, the template field points
   to a source in the same block of the concat where the `forcing` is
   mentioned.

This is a means to provide the model with space and time information
during training and inference.

The following fields are available:

latitude
   Each grid point has the value of its latitude in degrees.

   Range of values: :math:`[-90, 90]`

cos_latitude
   Each grid point has the value of :math:`cos(latitude/180*\pi)`.
   Unlike the **latitude** field, this field is periodic in space.

   Range of values: :math:`[-1, 1]`

sin_latitude
   Each grid point has the value of :math:`sin(latitude/180*\pi)`.
   Unlike the **latitude** field, this field is periodic in space.

   Range of values: :math:`[-1, 1]`

longitude
   Each grid point has the value of its longitude in degrees. Currently,
   the longitude is not normalised, and the values are the ones provided
   by the source used as a template.

   Range of values: :math:`[-180, 360)`

cos_longitude
   Each grid point has the value of :math:`cos(longitude/180*\pi)`.
   Unlike the **longitude** field, this field is periodic in space.

   Range of values: :math:`[-1, 1]`

sin_longitude
   Each grid point has the value of :math:`sin(longitude/180*\pi)`.
   Unlike the **longitude** field, this field is periodic in space.

   Range of values: :math:`[-1, 1]`

ecef_x
   Each grid point has the value of the **x** coordinate of the
   `Earth-Centred, Earth-Fixed (ECEF) coordinate system <ECEF>`_. The
   Earth is assumed to be a perfect sphere with a radius of 1.

   Range of values: :math:`[-1, 1]`

ecef_y
   Each grid point has the value of the **y** coordinate of the
   `Earth-Centred, Earth-Fixed (ECEF) coordinate system <ECEF>`_. The
   Earth is assumed to be a perfect sphere with a radius of 1.

   Range of values: :math:`[-1, 1]`

ecef_z
   Each grid point has the value of the **z** coordinate of the
   `Earth-Centred, Earth-Fixed (ECEF) coordinate system <ECEF>`_. The
   Earth is assumed to be a perfect sphere with a radius of 1.

   Range of values: :math:`[-1, 1]`

julian_day
   The Julian day is the fractional number of days since the 1st of
   January at 00:00 of the current year. For example, the Julian day of
   1st of January at 12:00 is 0.5. Every grid point has the same value
   of the Julian day at the given date.

   Range of values: :math:`[0, 365)` on a non-leap year and :math:`[0,
   366)` on a leap year

cos_julian_day
   Each grid point has the value of
   :math:`cos(julian\_day/365.25*2*\pi)`. Unlike the **julian_day**
   field, this field is periodic in time.

   Range of values: :math:`[-1, 1]`

sin_julian_day
   Each grid point has the value of
   :math:`sin(julian\_day/365.25*2*\pi)`. Unlike the **julian_day**
   field, this field is periodic in time.

   Range of values: :math:`[-1, 1]`

local_time
   Each grid point has the value of the local time in hours. The
   computation of the local time is solely based on the longitude of the
   grid point (i.e. no time zone information is used). The local time is
   computed as the fractional part of the longitude in hours, starting
   from 0 at the Greenwich meridian. So, for example, a grid point with
   a longitude of 0 will have a local time of 12.5 at 12:30 UTC.

   Range of values: :math:`[0, 24)`

cos_local_time
   Each grid point has the value of :math:`cos(local\_time/24*2*\pi)`.
   Unlike the **julian_day** field, this field is periodic in time.

   Range of values: :math:`[-1, 1]`

sin_local_time
   Each grid point has the value of :math:`sin(local\_time/24*2*\pi)`.
   Unlike the **julian_day** field, this field is periodic in time.

   Range of values: :math:`[-1, 1]`

insolation
   This is an alias for the **cos_solar_zenith_angle** field.

   Range of values: :math:`[?, ?]`

cos_solar_zenith_angle
   This is an alias for the **insolation** field. See earthkit.meteo_
   for more information.

   Range of values: :math:`[?, ?]`

toa_incident_solar_radiation
   Top of atmosphere incident solar radiation. See earthkit.meteo_ for
   more information.

   Range of values: :math:`[?, ?]`

.. _earthkit.meteo: https://github.com/ecmwf/earthkit-meteo/blob/74654e0b188e5a201d8268e93376246c925e3172/earthkit/meteo/solar/__init__.py#L49C4-L49C27

.. _ecef: https://en.wikipedia.org/wiki/Earth-centred,_Earth-fixed_coordinate_system
