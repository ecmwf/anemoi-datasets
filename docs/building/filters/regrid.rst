########
 regrid
########

When building a dataset for a specific model, it is possible that the
source grid or resolution does not fit the needs. In that case, it is
possible to add a filter to interpolate the data to a target grid. The
filter is part of the ``anemoi-transform`` package. It will call the
``interpolate`` function from `earthkit-regrid
<https://earthkit-regrid.readthedocs.io/en/latest/interpolate.html>`_ if
the keys ``method``, ``in_grid`` and ``out_grid`` are provided and if a
`pre-generated matrix
<https://earthkit-regrid.readthedocs.io/en/latest/inventory/index.html>`_
exists for this transformation. Otherwise, it is possible to provide a
``regrid matrix`` previously generated with ``anemoi-transform
make-regrid-matrix``. The generated matrix is an NPZ file containing the
input/output coordinates, the indices, and the weights of the
interpolation.

``regrid`` is a :ref:`filter <filters>` that must follow a :ref:`source
<sources>` or another filter in a :ref:`building-pipe` operation.

.. literalinclude:: yaml/regrid1.yaml
   :language: yaml

.. literalinclude:: yaml/regrid2.yaml
   :language: yaml
