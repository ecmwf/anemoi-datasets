.. _configuration:

###############
 Configuration
###############

When the ``open_dataset`` function is called with a string that does
not ends with ``.zarr`` or ``.zip``, it is considered a dataset name and not
a path or a URL.

In that case, the *Anemoi* configuration is read from ``~/.anemoi.toml``. Below is an
example of such a configuration:

.. literalinclude:: configuration.toml
   :language: toml
