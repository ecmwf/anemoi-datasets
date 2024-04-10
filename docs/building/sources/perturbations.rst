.. _perturbations:

###############
 perturbations
###############

Perturbations refers to the small variations centered around a nominal
value of a parameter. When dealing with ensemble forecasting, the
perturbations are related to the difference between `ensemble members`
and their given center
(https://www.ecmwf.int/en/elibrary/75394-ensemble-forecasting).

The `perturbations` function computes a set of ensemble members centered
on a difference center from previous ensemble members using the
following formula:

.. math::

   members_{new} = center + ( members - \overline{members} )

It uses the following arguments:

ensembles
   A pointer to the ensemble members.

center
   A pointer to the new center requested.

.. literalinclude:: yaml/perturbations.yaml
   :language: yaml
