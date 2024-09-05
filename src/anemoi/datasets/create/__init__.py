# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os

LOG = logging.getLogger(__name__)


def _ignore(*args, **kwargs):
    pass


class Creator:
    def __init__(
        self,
        path,
        config=None,
        cache=None,
        use_threads=False,
        statistics_tmp=None,
        overwrite=False,
        test=None,
        progress=None,
        **kwargs,
    ):
        self.path = path  # Output path
        self.config = config
        self.cache = cache
        self.use_threads = use_threads
        self.statistics_tmp = statistics_tmp
        self.overwrite = overwrite
        self.test = test
        self.progress = progress if progress is not None else _ignore
        self.kwargs = kwargs

    def check_unkown_kwargs(self):
        for k in self.kwargs:
            raise Exception(f"Unknown kwargs: {self.kwargs}")

    def init(self):
        check_name = self.kwargs.pop("check_name", False)
        self.check_unkown_kwargs()

        # check path
        _, ext = os.path.splitext(self.path)
        assert ext != "zarr", f"Unsupported extension={ext}"
        if self._path_readable() and not self.overwrite:
            raise Exception(f"{self.path} already exists. Use overwrite=True to overwrite.")

        from .loaders import InitialiserLoader

        with self._cache_context():
            obj = InitialiserLoader.from_config(
                path=self.path,
                config=self.config,
                statistics_tmp=self.statistics_tmp,
                use_threads=self.use_threads,
                progress=self.progress,
                test=self.test,
            )
            return obj.initialise(check_name=check_name)

    def load(self):
        parts = self.kwargs.pop("parts", None)
        from .loaders import ContentLoader

        with self._cache_context():
            loader = ContentLoader.from_dataset_config(
                path=self.path,
                statistics_tmp=self.statistics_tmp,
                use_threads=self.use_threads,
                progress=self.progress,
                parts=parts,
            )
            loader.load()

    def statistics(self):
        output = self.kwargs.pop("output", None)
        start = self.kwargs.pop("start", None)
        end = self.kwargs.pop("end", None)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAdder

        loader = StatisticsAdder.from_dataset(
            path=self.path,
            use_threads=self.use_threads,
            progress=self.progress,
            statistics_tmp=self.statistics_tmp,
            statistics_output=output,
            recompute=False,
            statistics_start=start,
            statistics_end=end,
        )
        loader.run()
        assert loader.ready()

    def size(self):
        from .loaders import DatasetHandler
        from .size import compute_directory_sizes

        metadata = compute_directory_sizes(self.path)
        handle = DatasetHandler.from_dataset(path=self.path, use_threads=self.use_threads)
        handle.update_metadata(**metadata)

    def cleanup(self):
        from .loaders import DatasetHandlerWithStatistics

        cleaner = DatasetHandlerWithStatistics.from_dataset(
            path=self.path, use_threads=self.use_threads, progress=self.progress, statistics_tmp=self.statistics_tmp
        )
        cleaner.tmp_statistics.delete()
        cleaner.registry.clean()

    def patch(self):
        from .patch import apply_patch

        apply_patch(self.path, **self.kwargs)

    def init_additions(self):
        delta = self.kwargs.pop("delta", [])
        recompute_statistics = self.kwargs.pop("recompute_statistics", False)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAddition
        from .loaders import TendenciesStatisticsAddition
        from .loaders import TendenciesStatisticsDeltaNotMultipleOfFrequency

        if recompute_statistics:
            a = StatisticsAddition.from_dataset(path=self.path, use_threads=self.use_threads)
            a.initialise()

        for d in delta:
            try:
                a = TendenciesStatisticsAddition.from_dataset(
                    path=self.path, use_threads=self.use_threads, progress=self.progress, delta=d
                )
                a.initialise()
            except TendenciesStatisticsDeltaNotMultipleOfFrequency:
                LOG.info(f"Skipping delta={d} as it is not a multiple of the frequency.")

    def run_additions(self):
        delta = self.kwargs.pop("delta", [])
        recompute_statistics = self.kwargs.pop("recompute_statistics", False)
        parts = self.kwargs.pop("parts", None)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAddition
        from .loaders import TendenciesStatisticsAddition
        from .loaders import TendenciesStatisticsDeltaNotMultipleOfFrequency

        if recompute_statistics:
            a = StatisticsAddition.from_dataset(path=self.path, use_threads=self.use_threads)
            a.run(parts)

        for d in delta:
            try:
                a = TendenciesStatisticsAddition.from_dataset(
                    path=self.path, use_threads=self.use_threads, progress=self.progress, delta=d
                )
                a.run(parts)
            except TendenciesStatisticsDeltaNotMultipleOfFrequency:
                LOG.debug(f"Skipping delta={d} as it is not a multiple of the frequency.")

    def finalise_additions(self):
        delta = self.kwargs.pop("delta", [])
        recompute_statistics = self.kwargs.pop("recompute_statistics", False)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAddition
        from .loaders import TendenciesStatisticsAddition
        from .loaders import TendenciesStatisticsDeltaNotMultipleOfFrequency

        if recompute_statistics:
            a = StatisticsAddition.from_dataset(path=self.path, use_threads=self.use_threads)
            a.finalise()

        for d in delta:
            try:
                a = TendenciesStatisticsAddition.from_dataset(
                    path=self.path, use_threads=self.use_threads, progress=self.progress, delta=d
                )
                a.finalise()
            except TendenciesStatisticsDeltaNotMultipleOfFrequency:
                LOG.debug(f"Skipping delta={d} as it is not a multiple of the frequency.")

    def finalise(self):
        self.statistics()
        self.size()

    def create(self):
        self.init()
        self.load()
        self.finalise()
        self.additions()
        self.cleanup()

    def additions(self):
        #from .loaders import read_temporary_config_from_dataset
        #config = read_temporary_config_from_dataset(self.path)

        self.init_additions()
        self.run_additions()
        self.finalise_additions()

    def _cache_context(self):
        from .utils import cache_context

        return cache_context(self.cache)

    def _path_readable(self):
        import zarr

        try:
            zarr.open(self.path, "r")
            return True
        except zarr.errors.PathNotFoundError:
            return False

    def verify(self):
        from .loaders import DatasetVerifier

        handle = DatasetVerifier.from_dataset(path=self.path, use_threads=self.use_threads)

        handle.verify()
