.. _recentre:

##########
 recentre
##########

Perturbations refer to the small variations centred around a nominal
value of a parameter. When dealing with `ensemble forecasting`_, the
perturbations are related to the difference between `ensemble members`
and their given `centre`.

The `recentre` function computes a set of new ensemble members centred
on a different centre from previous ensemble members using the following
formula:

.. math::

   members_{new} = centre + ( members - \overline{members} )

Additionally, some variables must be non-negative to have a physical
meaning (e.g. accumulated variables or `specific humidity`). To ensure
this, positive clipping is performed using the alternative formula:

.. math::

   members_{new} = max(0, centre + ( members - \overline{members} ))

The current implementation enforces that the following variables are
positive when using the `perturbations` function:

+----------+------------------------------+
| Variable | Description                  |
+==========+==============================+
| q        | `Specific humidity`_         |
+----------+------------------------------+
| cp       | `Convective precipitation`_  |
+----------+------------------------------+
| lsp      | `Large-scale precipitation`_ |
+----------+------------------------------+
| tp       | `Total precipitation`_       |
+----------+------------------------------+

It uses the following arguments:

members
   A :ref:`reference <yaml-reference>` to the ensemble members.

centre
   A :ref:`reference <yaml-reference>` to the new centre requested.

Examples

.. literalinclude:: yaml/recentre.yaml
   :language: yaml

.. _convective precipitation: https://codes.ecmwf.int/grib/param-db/?id=143

.. _ensemble forecasting: https://www.ecmwf.int/en/elibrary/75394-ensemble-forecasting

.. _large-scale precipitation: https://codes.ecmwf.int/grib/param-db/?id=142

.. _specific humidity: https://codes.ecmwf.int/grib/param-db/?id=133

.. _total precipitation: https://codes.ecmwf.int/grib/param-db/?id=228
