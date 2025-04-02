#########
 wz_to_w
#########

The ``wz_to_w`` filter converts geometric vertical velocity (provided in
m/s) to vertical velocity in pressure coordinates (Pa/s). This filter
must follow a source that provides geometric vertical velocity.
Geometric vertical velocity is removed by the filter, and pressure
vertical velocity is added.

.. literalinclude:: yaml/wz_to_w.yaml
   :language: yaml
