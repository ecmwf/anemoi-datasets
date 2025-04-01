#####
 sum
#####

The ``sum`` filter computes the sum over multiple variables. This can be
useful for computing total precipitation from its components (snow,
rain) or summing the components of total column-integrated water. This
filter must follow a source that provides the list of variables to be
summed up. These variables are removed by the filter and replaced by a
single summed variable.

.. literalinclude:: yaml/sum.yaml
   :language: yaml
