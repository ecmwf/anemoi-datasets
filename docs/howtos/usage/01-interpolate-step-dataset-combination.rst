.. _usage_interpolate_step_dataset_combination:

########################################################
 # Combine datasets with different timestep frequencies
########################################################

Here we explain how to combine two existing datasets with different
timestep frequencies. In this example we consider two datasets :
`dataset1` with a te,mporal frequency of 3h and `dataset2` with a
temporal frequency of 24h. The goal is to combine the two datasets into
a single dataset with a temporal frequency of 3h or 24h. We consider two
cases, in case one we would like to bring the larger timestep dataset to
the smaller timestep dataset, in case two we would like to bring the
smaller timestep dataset to the larger timestep dataset.

*********************************
 Interpolate to higher frequency
*********************************

In this case we will use the `interpolate_frequency` option to bring
`dataset2` to the 3h timestep of dataset1.

.. literalinclude:: yaml/interpolate_frequency

The `adjust_dates` is in case the end or start dates do not exactly
match.

***************************
 Sample to lower frequency
***************************

This case is straightforward, we will can just specify the required 24h
frequency for datset1.

.. literalinclude:: yaml/sample
