This source uses `Xarray`_ internally to  access the data, and assumes
that it follows the `CF conventions`_.

You specify any valid `xarray.open_dataset`_ arguments in the source.

.. literalinclude:: yaml/xarray-based.yaml
    :language: yaml


.. _Xarray: https://docs.xarray.dev/en/stable/index.html

.. _CF conventions: http://cfconventions.org/

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
