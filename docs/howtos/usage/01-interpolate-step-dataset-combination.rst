.. _interpolate-step:

##############################################
 Combining cutout with complementing datasets
##############################################

Here we explain how to combine a cutout with a complementing dataset.

*************************
 Interpolate to LAM grid
*************************

In this case, we will use the a ``lam_dataset`` in a different grid that
contains just one variable and a ``global_dataset``. What we want to do
is to interpolate the ``global_dataset`` to the resulting dataset from
the cutout grid operation.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
      dataset={
      'complement':
         {
         "cutout": ["/home/ecm2559/dev_perm/anemoi-ecmwf-rodeo-benchmark/opera_2km_dataset/anemoi-datasets-configs/datasets/opera_6hr_2015_cutout_96.zarr",
                    {"dataset": "/home/mlx/ai-ml/datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6.zarr", "select": "variable"}
                    ],
         "min_distance_km": 1,
         "plot": "prefix",
         "adjust": "dates",
      },
      'source': {
         'dataset':"/home/mlx/ai-ml/datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6.zarr",
      },
    'interpolation':'nearest'},
      start="2015-01-01",
      end="2015-02-01",
   )

or for the config file:

.. code:: yaml

   dataset:
   complement:
      cutout:
         - dataset: lam-dataset
         - dataset: global-dataset
           select: variable # this is the variable we select to match the LAM dataset
      min_distance_km: 1
      adjust: dates
   source:
      - dataset: global-dataset
   interpolation: nearest
    start: 2015-01-01
    end: 2015-02-01
    drop: []

The ``adjust`` option is in case the end or start dates do not exactly
match.

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
