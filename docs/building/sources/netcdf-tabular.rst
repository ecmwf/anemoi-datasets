################
 netcdf-tabular
################

This source reads point observations (for example satellite radiances or
in-situ measurements) from NetCDF files using ``xarray``. Each file must
hold one-dimensional variables that share a single "point" dimension,
with one latitude, one longitude and one time value per point.

The files are decoded following the CF conventions: ``scale_factor``,
``add_offset`` and ``_FillValue`` are applied (fill values become NaN),
and the time variable is decoded from its ``units`` attribute, so every
file may use a different time reference (e.g. ``seconds since
2023-12-01 12:19:55``).

The ``path`` parameter accepts a glob pattern or a list of them. Only the
files whose observation times overlap the requested dates are read; the
selection uses the times stored in the files, not their names.

The ``flavour`` parameter gives the names of the ``latitude``,
``longitude`` and ``date`` variables in the files. The defaults are
``latitude``, ``longitude`` and ``time``.

The ``variables`` parameter selects the variables to read. By default,
all the variables along the point dimension are read, except the
latitude, longitude and time ones.

.. literalinclude:: yaml/netcdf-tabular.yaml
   :language: yaml
