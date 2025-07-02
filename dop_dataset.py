import json
import os
import random
from typing import Optional

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from anemoi.datasets import open_dataset

CONFIG = dict(
    data=dict(
        # era5=dict(
        #     dataset=dict(dataset="aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8", set_group="era5"),
        #     # preprocessors=dict(
        #     #    tp=[dict(normalizer="mean-std")]),
        #     # ),
        # ),
        snow=dict(dataset="observations-testing-2018-2018-6h-v1-one-month"),
        metop_a=dict(dataset="observations-testing-2018-2018-6h-v1-one-month"),
        amsr2_h180=dict(dataset="observations-testing-2018-2018-6h-v1-one-month"),
    ),
    sample=dict(
        GROUPS=dict(
            input=dict(
                GROUPS=dict(
                    # fields=dict(  # "fields" is a user defined key
                    #     STEPS=dict(
                    #         _6h=dict(
                    #             variables=["q_50", "2t"],
                    #             data="era5",
                    #         ),
                    #         _0h=dict(
                    #             variables=["q_50", "2t"],
                    #             data="era5",
                    #         ),
                    #     ),
                    # ),
                    # user-friendly config would be:
                    # fields=dict(
                    #     steps=['-6h', '0h'],
                    #     variables=["q_50", "2t"],
                    #     data="era5",
                    # ),
                    ascat_metop_a=dict(  # "metar" is a user defined key
                        STEPS=dict(
                            _6h=dict(
                                variables=["scatss_1", "scatss_2"],
                                data="metop_a",
                            ),
                        ),
                    ),
                    snow=dict(  # "iasi" is a user defined key
                        STEPS=dict(
                            _6h=dict(
                                variables=["sdepth_0"],
                                data="snow",
                            ),
                        ),
                    ),
                    amsr2=dict(  # "iasi" is a user defined key
                        STEPS=dict(
                            _6h=dict(
                                variables=["rawbt_1", "rawbt_2", "rawbt_3", "rawbt_4"],
                                data="amsr_h180",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


class Sample:
    def __init__(self, datahandlers):
        self.datahandlers = datahandlers

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self._build_tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def _build_tree(self, label="Sample"):
        return Tree(label)


class GroupedSample(Sample):
    def __init__(self, datahandlers, dic):
        super().__init__(datahandlers)
        self._samples = {k: sample_factory(**v) for k, v in dic.items()}

    def __getitem__(self, item):
        return {k: v[item] for k, v in self._samples.items()}

    def _build_tree(self, label="GroupedSample"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class StepSample(Sample):
    def __init__(self, datahandlers, dic):
        super().__init__(datahandlers)
        self._samples = {k: sample_factory(**v) for k, v in dic.items()}

    def __getitem__(self, item):
        out = []
        for k, v in self._samples.items():
            if k == "_6h":
                out.append(v[item - 1])
            elif k == "_0h":
                out.append(v[item])
            elif k == "p6h":
                out.append(v[item + 1])
        return out

    def _build_tree(self, label="GroupedSample"):
        tree = Tree(label)
        for k, v in self._samples.items():
            subtree = v._build_tree(label=f"{k}: {type(v).__name__}")
            tree.add(subtree)
        return tree


class Leaf(Sample):
    def __init__(self, datahandlers, variables, data):
        super().__init__(datahandlers)
        self.data_key = data
        self.variables = variables

    def __getitem__(self, item):
        result = Result(self.data_key, item, variables=self.variables)
        return result.load()

    def _build_tree(self, label="Leaf"):
        return Tree(f"{label}  -> {self.data_key} variables={self.variables}")


def sample_factory(datahandlers=None, **kwargs):
    kwargs = kwargs.copy()
    if datahandlers is None:
        datahandlers = []
    if "GROUPS" in kwargs:
        return GroupedSample(datahandlers, kwargs["GROUPS"])
    if "STEPS" in kwargs:
        return StepSample(datahandlers, kwargs["STEPS"])
    if "variables" in kwargs:
        return Leaf(datahandlers, variables=kwargs["variables"], data=kwargs["data"])
    assert False, f"Unknown sample type for kwargs {kwargs}"


class Result:
    def __init__(self, datahandler_key, *args, variables=[], **kwargs):
        cfg = CONFIG["data"][datahandler_key]
        assert "select" not in cfg, (cfg, variables)
        variables = [f"{datahandler_key}.{v}" for v in variables]
        dh = DataHandler(datahandler_key, **cfg, select=variables)

        self.func = dh.__getitem__
        self.args = args
        self.kwargs = kwargs

    def load(self):
        return self.func(*self.args, **self.kwargs)

    def __repr__(self):
        inside = []
        inside += [str(arg) for arg in self.args]
        inside += [f"{k}={v}" for k, v in self.kwargs.items()]
        return f"Result({self.datahandler}  ({', '.join(inside)})"


class DataHandler:
    def __init__(self, name, **config):
        self.name = name
        if isinstance(config, str):
            config = dict(dataset=config)
        if isinstance(config["dataset"], str):
            config = dict(dataset=config)

        self.config = config
        self._config_str = " ".join(f"{k}={v}" for k, v in config.items())

    def is_grouped_dataset(self, ds):
        from anemoi.datasets.data.records import BaseRecordsDataset

        return isinstance(ds, BaseRecordsDataset)

    @property
    def ds(self):
        ds = open_dataset(**self.config["dataset"])
        print(f"🔍 Opened dataset {self.name} with config: {self._config_str}")
        if self.name not in ds.groups:
            raise ValueError(f"Group '{self.name}' not found in dataset. Available groups: {ds.groups}")
        ds = ds[self.name]
        print(f"   Available variables for group '{self.name}': {ds.variables}")
        return ds

    def __getitem__(self, item):
        data = self.ds[item]
        assert isinstance(data, np.ndarray), f"Expected np.array, got {type(data)}, {type(self.ds)}"
        return data
        return f"np.array ds[{item}] with ds from {self._config_str} "

    def __str__(self):
        return f"DataHandler({self._config_str})"


def show_yaml(structure):
    return yaml.dump(structure, indent=2, sort_keys=False)


def show_json(structure):
    return json.dumps(structure, indent=2, default=shorten_numpy)


def shorten_numpy(structure):
    if isinstance(structure, np.ndarray):
        return f"np.array({structure.shape})"
    return structure


def get_base_seed():
    """Get a base seed for random number generation.
    This is a placeholder function; replace with actual logic to get a base seed.
    """
    return 42  # Example fixed seed, replace with actual logic as needed


class DOPDataset(IterableDataset):
    def __init__(
        self,
        # config: dict,
        shuffle: bool = True,
        rollout: int = 1,
        multistep: int = 1,
        task: str = "training",
    ) -> None:

        self.shuffle = shuffle
        # self.config = config
        self.rollout = rollout
        self.multistep = multistep
        self.task = task

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle
        self.rng: Optional[np.random.Generator] = None
        self.worker_id: int = -1

        # "full" shuffling
        self.data_indices: Optional[np.ndarray] = None

        self.seed_comm_group_id = 0
        self.seed_comm_num_groups = 1

        self._sample_factory = sample_factory(**CONFIG["sample"])

        self.len = 25  # len(self._sample_factory)

    def __get_sample(self, index: int):
        """Get a sample from the dataset."""
        return self._sample_factory[index]

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID
        """
        self.worker_id = worker_id

        # Total number of valid ICs is dataset length minus rollout minus additional multistep inputs
        len_corrected = self.len - self.rollout - self.multistep + 1
        self.data_indices = np.arange(len_corrected, dtype=np.uint32)

        # Divide this equally across shards (one shard per group!)
        shard_size = len_corrected // self.seed_comm_num_groups
        shard_start = self.seed_comm_group_id * shard_size
        shard_end = min((self.seed_comm_group_id + 1) * shard_size, self.len - self.rollout - self.multistep + 1)

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        seed = get_base_seed()  # all workers get the same seed (so they all get the same index shuffle)
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)
        print("Sanity check random number:", sanity_rnd)

    def __iter__(self):
        if self.shuffle:
            # do a full shuffle, then get my index range
            shuffled_data_indices = self.rng.choice(self.data_indices, size=len(self.data_indices), replace=False)
            shuffled_chunk_indices = shuffled_data_indices[self.chunk_index_range]

            while True:  # the pl.Trainer will break out of this loop after a fixed number of samples
                idx = self.rng.choice(shuffled_chunk_indices)
                print(
                    f"TRAINING: Worker {self.worker_id} (pid {os.getpid()}) fetching sample index {idx} ...",
                )
                yield self.__get_sample(idx)

        else:
            shuffled_chunk_indices = self.data_indices[self.chunk_index_range]
            # no shuffle, just iterate over the chunk indices
            for idx in self.chunk_index_range:
                print(
                    f"VALIDATION: Worker {self.worker_id} (pid {os.getpid()}) fetching sample index {idx} ...",
                )
                yield self.__get_sample(idx)


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None
    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        print("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )


if __name__ == "__main__":

    ds = DOPDataset(
        # CONFIG,
        shuffle=False,
        rollout=1,
        multistep=1,
        task="training",
    )

    loader_params = {
        "batch_size": 1,  # must be 1 for the time being
        "batch_sampler": None,
        "num_workers": 2,
        "pin_memory": False,
        "worker_init_fn": worker_init_func,
        # "collate_fn": None, # collator_wrapper(return_original_metadata=cfg_.dataloader.return_dates),
    }

    dl = torch.utils.data.DataLoader(ds, **loader_params, sampler=None)

    for batch_idx, batch in enumerate(dl):
        print.info("%s", batch)
        if batch_idx >= 1:
            break
