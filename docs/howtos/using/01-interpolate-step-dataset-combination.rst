.. _interpolate-step:

######################################################
 Combine datasets with different timestep frequencies
######################################################

Here we explain how to combine two existing datasets with different
timestep frequencies. In this example, we consider two datasets:
``dataset-3h`` with an inherent temporal frequency of 3h and
``dataset-24h`` with an inherent temporal frequency of 24h. The goal is
to combine the two datasets into a dataset with a temporal frequency of
either 3h or 24h.

*********************************
 Interpolate to higher frequency
*********************************

In this case, we will use the ``interpolate_frequency`` option to bring
``dataset-24h`` to the 3h timestep of ``dataset-3h``.

.. literalinclude:: code/interpolate1.py
   :language: python

or in the config file:

.. literalinclude:: yaml/interpolate1.yaml
   :language: yaml

The ``adjust`` option is in case the end or start dates do not exactly
match.

***************************
 Sample to lower frequency
***************************

This case is straightforward; we can just specify the required 24h
frequency for ``dataset-3h``.

.. literalinclude:: code/interpolate2.py
   :language: python

or for the config file:

.. literalinclude:: yaml/interpolate2.yaml
   :language: yaml
