# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import logging
import os
from copy import deepcopy

import yaml
from anemoi.utils.config import DotDict
from anemoi.utils.config import load_any_dict_format
from earthkit.data.core.order import normalize_order_by

from anemoi.datasets.dates.groups import Groups

LOG = logging.getLogger(__name__)


def _get_first_key_if_dict(x):
    if isinstance(x, str):
        return x
    return list(x.keys())[0]


def ensure_element_in_list(lst, elt, index):
    if elt in lst:
        assert lst[index] == elt
        return lst

    _lst = [_get_first_key_if_dict(d) for d in lst]
    if elt in _lst:
        assert _lst[index] == elt
        return lst

    return lst[:index] + [elt] + lst[index:]


def check_dict_value_and_set(dic, key, value):
    if key in dic:
        if dic[key] == value:
            return
        raise ValueError(f"Cannot use {key}={dic[key]}. Must use {value}.")
    LOG.info(f"Setting {key}={value} in config")
    dic[key] = value


def resolve_includes(config):
    if isinstance(config, list):
        return [resolve_includes(c) for c in config]
    if isinstance(config, dict):
        include = config.pop("<<", {})
        new = deepcopy(include)
        new.update(config)
        return {k: resolve_includes(v) for k, v in new.items()}
    return config


class Config(DotDict):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, str):
            self.config_path = os.path.realpath(config)
            config = load_any_dict_format(config)
        else:
            config = deepcopy(config if config is not None else {})
        config = resolve_includes(config)
        config.update(kwargs)
        super().__init__(config)


class OutputSpecs:
    def __init__(self, config, parent):
        self.config = config
        if "order_by" in config:
            assert isinstance(config.order_by, dict), config.order_by

        self.parent = parent

    @property
    def dtype(self):
        return self.config.dtype

    @property
    def order_by_as_list(self):
        # this is used when an ordered dict is not supported (e.g. zarr attributes)
        return [{k: v} for k, v in self.config.order_by.items()]

    def get_chunking(self, coords):
        user = deepcopy(self.config.chunking)
        chunks = []
        for k, v in coords.items():
            if k in user:
                chunks.append(user.pop(k))
            else:
                chunks.append(len(v))
        if user:
            raise ValueError(
                f"Unused chunking keys from config: {list(user.keys())}, not in known keys : {list(coords.keys())}"
            )
        return tuple(chunks)

    @property
    def order_by(self):
        return self.config.order_by

    @property
    def remapping(self):
        return self.config.remapping

    @property
    def flatten_grid(self):
        return self.config.flatten_grid

    @property
    def statistics(self):
        return self.config.statistics


class LoadersConfig(Config):
    def __init__(self, config, *args, **kwargs):

        super().__init__(config, *args, **kwargs)

        # TODO: should use a json schema to validate the config

        self.setdefault("dataset_status", "experimental")
        self.setdefault("description", "No description provided.")
        self.setdefault("licence", "unknown")
        self.setdefault("attribution", "unknown")

        self.setdefault("build", Config())
        self.build.setdefault("group_by", "monthly")
        self.build.setdefault("use_grib_paramid", False)
        self.build.setdefault("variable_naming", "default")
        variable_naming = dict(
            param="{param}",
            param_levelist="{param}_{levelist}",
            default="{param}_{levelist}",
        ).get(self.build.variable_naming, self.build.variable_naming)

        self.setdefault("output", Config())
        self.output.setdefault("order_by", ["valid_datetime", "param_level", "number"])
        self.output.setdefault("remapping", Config(param_level=variable_naming))
        self.output.setdefault("statistics", "param_level")
        self.output.setdefault("chunking", Config(dates=1, ensembles=1))
        self.output.setdefault("dtype", "float32")

        if "statistics_start" in self.output:
            raise ValueError("statistics_start is not supported anymore. Use 'statistics:start:' instead")
        if "statistics_end" in self.output:
            raise ValueError("statistics_end is not supported anymore. Use 'statistics:end:' instead")

        self.setdefault("statistics", Config())
        if "allow_nans" not in self.statistics:
            self.statistics.allow_nans = []

        check_dict_value_and_set(self.output, "flatten_grid", True)
        check_dict_value_and_set(self.output, "ensemble_dimension", 2)

        assert isinstance(self.output.order_by, (list, tuple)), self.output.order_by
        self.output.order_by = ensure_element_in_list(self.output.order_by, "number", self.output.ensemble_dimension)

        order_by = self.output.order_by
        assert len(order_by) == 3, order_by
        assert _get_first_key_if_dict(order_by[0]) == "valid_datetime", order_by
        assert _get_first_key_if_dict(order_by[2]) == "number", order_by

        self.output.order_by = normalize_order_by(self.output.order_by)

        self.dates["group_by"] = self.build.group_by

        ###########

        self.reading_chunks = self.get("reading_chunks")

    def get_serialisable_dict(self):
        return _prepare_serialisation(self)


def _prepare_serialisation(o):
    if isinstance(o, dict):
        dic = {}
        for k, v in o.items():
            v = _prepare_serialisation(v)
            if k == "order_by" and isinstance(v, dict):
                # zarr attributes are saved with sort_keys=True
                # and ordered dict are reordered.
                # This is a problem for "order_by"
                # We ensure here that the order_by key contains
                # a list of dict
                v = [{kk: vv} for kk, vv in v.items()]
            dic[k] = v
        return dic

    if isinstance(o, (list, tuple)):
        return [_prepare_serialisation(v) for v in o]

    if o in (None, True, False):
        return o

    if isinstance(o, (str, int, float)):
        return o

    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

    return str(o)


def set_to_test_mode(cfg):
    NUMBER_OF_DATES = 4

    dates = cfg["dates"]
    LOG.warning(f"Running in test mode. Changing the list of dates to use only {NUMBER_OF_DATES}.")
    groups = Groups(**LoadersConfig(cfg).dates)

    dates = groups.dates
    cfg["dates"] = dict(
        start=dates[0],
        end=dates[NUMBER_OF_DATES - 1],
        frequency=dates.frequency,
        group_by=NUMBER_OF_DATES,
    )

    def set_element_to_test(obj):
        if isinstance(obj, (list, tuple)):
            for v in obj:
                set_element_to_test(v)
            return
        if isinstance(obj, (dict, DotDict)):
            if "grid" in obj:
                previous = obj["grid"]
                obj["grid"] = "20./20."
                LOG.warning(f"Running in test mode. Setting grid to {obj['grid']} instead of {previous}")
            if "number" in obj:
                if isinstance(obj["number"], (list, tuple)):
                    previous = obj["number"]
                    obj["number"] = previous[0:3]
                    LOG.warning(f"Running in test mode. Setting number to {obj['number']} instead of {previous}")
            for k, v in obj.items():
                set_element_to_test(v)
            if "constants" in obj:
                constants = obj["constants"]
                if "param" in constants and isinstance(constants["param"], list):
                    constants["param"] = ["cos_latitude"]

    set_element_to_test(cfg)


def loader_config(config, is_test=False):
    config = Config(config)
    if is_test:
        set_to_test_mode(config)
    obj = LoadersConfig(config)

    # yaml round trip to check that serialisation works as expected
    copy = obj.get_serialisable_dict()
    copy = yaml.load(yaml.dump(copy), Loader=yaml.SafeLoader)
    copy = Config(copy)
    copy = LoadersConfig(config)

    a = yaml.dump(obj)
    b = yaml.dump(copy)
    if a != b:
        print(a)
        print(b)
        raise ValueError("Serialisation failed")

    return copy


def build_output(*args, **kwargs):
    return OutputSpecs(*args, **kwargs)
