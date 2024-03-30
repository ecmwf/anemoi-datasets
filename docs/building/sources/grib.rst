######
 grib
######

To read all the GRIB from a file, use the following:

.. literalinclude:: yaml/grib1.yaml
   :language: yaml

You can also read specific GRIB messages by specifying them using the
MARS language (excluding the keywords `date`, `time` and `step`, as well
as any post-processing options, such as `grid` or `area`):

.. literalinclude:: yaml/grib2.yaml
   :language: yaml

You can also read a collection of GRIB files, using Unix' shell
wildcards_ (see also :py:mod:`fnmatch`):

.. literalinclude:: yaml/grib3.yaml
   :language: yaml

You can also use the requested `date` to build the filenames.

.. literalinclude:: yaml/grib4.yaml
   :language: yaml

The patterns in between the curly brackets are replaced by the values of
the `date` and formatted according to the Python :py:mod:`datetime`'s
:py:meth:`strftime <datetime.datetime.strftime>` function.

.. note::

   You can combine all the above options when selecting GRIB messages
   from file.

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)
