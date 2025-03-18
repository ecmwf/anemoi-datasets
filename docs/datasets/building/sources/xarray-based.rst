This source uses Xarray_ internally to access the data, and assumes that
it follows the `CF conventions`_.

You specify any valid xarray.open_dataset_ arguments in the source.

.. literalinclude:: yaml/xarray-based.yaml
   :language: yaml

.. _cf conventions: http://cfconventions.org/

.. _xarray: https://docs.xarray.dev/en/stable/index.html

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
