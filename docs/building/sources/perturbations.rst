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

Additionally, for the `accumulated` variables, positive clipping is
performed to ensure that the final value of the variable is positive,
using the alternative fomula :

.. math::

   members_{new} = max(0, center + ( members - \overline{members} ))

The current implementation consider the following variables as
`accumulated` variables:

+-----+-----------------------------+-------------------------------------------------------+
|     | Variable                    | Description                                           |
+=====+=============================+=======================================================+
| q   | Specific humidity           | [Link](https://codes.ecmwf.int/grib/param-db/?id=133) |
+-----+-----------------------------+-------------------------------------------------------+
| cp  | Convective precipitation    | [Link](https://codes.ecmwf.int/grib/param-db/?id=143) |
+-----+-----------------------------+-------------------------------------------------------+
| lsp | Large-scale precipitation   | [Link](https://codes.ecmwf.int/grib/param-db/?id=142) |
+-----+-----------------------------+-------------------------------------------------------+
| tp  | Total precipitation         | [Link](https://codes.ecmwf.int/grib/param-db/?id=228) |
+-----+-----------------------------+-------------------------------------------------------+

It uses the following arguments:

ensembles
   A :ref:`reference <yaml-reference>` to the ensemble members.

center
   A :ref:`reference <yaml-reference>` to the new center requested.

.. literalinclude:: yaml/perturbations.yaml
   :language: yaml

.. _ensemble forecasting: https://www.ecmwf.int/en/elibrary/75394-ensemble-forecasting
