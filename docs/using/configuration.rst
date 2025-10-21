.. _configuration:

###############
 Configuration
###############

When the ``open_dataset`` function is called with a string that does not
end with ``.zarr`` or ``.zip``, it is considered a dataset name and not
a path or a URL.

In that case, the *Anemoi* configuration is read from
``~/.config/anemoi/settings.toml``. Below is an example of such a
configuration:

.. literalinclude:: configuration.toml
   :language: toml

Then, the name passed to ``open_dataset`` is used to look for a possible
path or URL:

-  If the name is listed in the ``[datasets.named]``, the corresponding
   path is used.
-  Otherwise, the suffix ``.zarr`` is added to the name, and the file is
   searched at every location listed in the ``path`` list.

See :ref:`miscellaneous` to modify the list of named datasets and the
path temporarily.
