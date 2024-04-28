# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import warnings
from copy import deepcopy

import numpy as np
from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output

from anemoi.datasets.create.check import check_data_values
from anemoi.datasets.create.functions import assert_is_fieldset
from anemoi.datasets.create.functions.actions.mars import mars


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    if isinstance(x, str):
        return x.split("/")
    return [x]


def normalise_number(number):
    number = to_list(number)

    if len(number) > 4 and (number[1] == "to" and number[3] == "by"):
        return list(range(int(number[0]), int(number[2]) + 1, int(number[4])))

    if len(number) > 2 and number[1] == "to":
        return list(range(int(number[0]), int(number[2]) + 1))

    return number


def normalise_request(request):
    request = deepcopy(request)
    if "number" in request:
        request["number"] = normalise_number(request["number"])
    if "time" in request:
        request["time"] = to_list(request["time"])
    request["param"] = to_list(request["param"])
    return request


def load_if_needed(context, dates, dict_or_dataset):
    if isinstance(dict_or_dataset, dict):
        dict_or_dataset = normalise_request(dict_or_dataset)
        dict_or_dataset = mars(context, dates, dict_or_dataset)
    return dict_or_dataset


def perturbations(context, dates, members, center, remapping={}, patches={}):
    members = load_if_needed(context, dates, members)
    center = load_if_needed(context, dates, center)

    keys = ["param", "level", "valid_datetime", "date", "time", "step", "number"]

    def check_compatible(f1, f2, ignore=["number"]):
        for k in keys + ["grid", "shape"]:
            if k in ignore:
                continue
            assert f1.metadata(k) == f2.metadata(k), (k, f1.metadata(k), f2.metadata(k))

    print(f"Retrieving ensemble data with {members}")
    print(f"Retrieving center data with {center}")

    members = members.order_by(*keys)
    center = center.order_by(*keys)

    number_list = members.unique_values("number")["number"]
    n_numbers = len(number_list)

    if len(center) * n_numbers != len(members):
        print(len(center), n_numbers, len(members))
        for f in members:
            print("Member: ", f)
        for f in center:
            print("Center: ", f)
        raise ValueError(f"Inconsistent number of fields: {len(center)} * {n_numbers} != {len(members)}")

    # prepare output tmp file so we can read it back
    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    for i, center_field in enumerate(center):
        param = center_field.metadata("param")

        # load the center field
        center_np = center_field.to_numpy()

        # load the ensemble fields and compute the mean
        members_np = np.zeros((n_numbers, *center_np.shape))

        for j in range(n_numbers):
            ensemble_field = members[i * n_numbers + j]
            check_compatible(center_field, ensemble_field)
            members_np[j] = ensemble_field.to_numpy()

        mean_np = members_np.mean(axis=0)

        for j in range(n_numbers):
            template = members[i * n_numbers + j]
            e = members_np[j]
            m = mean_np
            c = center_np

            assert e.shape == c.shape == m.shape, (e.shape, c.shape, m.shape)

            FORCED_POSITIVE = [
                "q",
                "cp",
                "lsp",
                "tp",
            ]  # add "swl4", "swl3", "swl2", "swl1", "swl0", and more ?

            x = c - m + e

            if param in FORCED_POSITIVE:
                warnings.warn(f"Clipping {param} to be positive")
                x = np.maximum(x, 0)

            assert x.shape == e.shape, (x.shape, e.shape)

            check_data_values(x, name=param)
            out.write(x, template=template)
            template = None

    out.close()

    from climetlab import load_source

    ds = load_source("file", path)
    assert_is_fieldset(ds)
    # save a reference to the tmp file so it is deleted
    # only when the dataset is not used anymore
    ds._tmp = tmp

    assert len(ds) == len(members), (len(ds), len(members))

    return ds


execute = perturbations
