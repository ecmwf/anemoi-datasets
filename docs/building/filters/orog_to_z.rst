###########
 orog_to_z
###########

The ``orog_to_z`` filter converts orography (in metres) to surface
geopotential height (m^2/s^2) using the equation:

.. math::

   z &= g \cdot \textrm{orog}\\
   g &= 9.80665\ m \cdot s^{-1}

This filter must follow a source that provides orography, which is
replaced by surface geopotential height.

.. literalinclude:: yaml/orog_to_z.yaml
   :language: yaml
