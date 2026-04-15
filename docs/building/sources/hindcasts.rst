###########
 hindcasts
###########

.. note::

   The `hindcasts` source is currently using the `mars` source
   internally. This will be changed in the future.

Hindcasts data, also known as reforecasts, are unique because they are
run for a specific day of the year (such as 1 January or 11 June) for
multiple years. So, for a given reference date like 2022-05-12, we can
have hindcasts for 2001-05-12, 2002-05-12, 2003-05-12, and so on. This
is useful in many cases. For more details, please refer to this ECMWF
documentation.

The `hindcasts` source has a special argument called `reference_year`,
which represents the year of the reference date. Based on the requested
valid datetime and on the `reference_year`, the `hindcasts` source will
calculate the `hdate`, `date`, and `time` appropriately.

For example, if the `reference_year` is 2022, then the data for
2002-05-12 will use data with `hdate=2002-05-12` and `date=2022-05-12`.

.. literalinclude:: yaml/hindcasts.yaml
   :language: yaml

Using `step` in the `hindcasts` source is implemented and works as
expected.
