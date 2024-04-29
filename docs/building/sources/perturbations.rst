.. _perturbations:

###############
 perturbations
###############

Perturbations refers to the small variations centered around a nominal
value of a parameter. When dealing with `ensemble forecasting`_, the
perturbations are related to the difference between `ensemble members`
and their given `center`.

The `perturbations` function computes a set of new ensemble members
centered on a different center from previous ensemble members using the
following formula:

.. math::

   members_{new} = center + ( members - \overline{members} )

Additionally, some variables must be non-negative to have a physical
meaning (e.g. accumulated variables or `specific humidity`). To ensure
this, positive clipping is performed using the alternative fomula :

.. math::

   members_{new} = max(0, center + ( members - \overline{members} ))

The current implementation enforces that following variables are
positive when using the `perturbations` function :

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

center
   A :ref:`reference <yaml-reference>` to the new center requested.

Examples

.. literalinclude:: yaml/perturbations.yaml
   :language: yaml

.. _convective precipitation: https://codes.ecmwf.int/grib/param-db/?id=143

.. _ensemble forecasting: https://www.ecmwf.int/en/elibrary/75394-ensemble-forecasting

.. _large-scale precipitation: https://codes.ecmwf.int/grib/param-db/?id=142

.. _specific humidity: https://codes.ecmwf.int/grib/param-db/?id=133

.. _total precipitation: https://codes.ecmwf.int/grib/param-db/?id=228
