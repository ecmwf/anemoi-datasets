# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from copy import deepcopy

from anemoi.datasets.compute.recentre import recentre as _recentre

from .mars import mars


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


def recentre(context, dates, members, centre, alpha=1.0, remapping={}, patches={}):
    members = load_if_needed(context, dates, members)
    centre = load_if_needed(context, dates, centre)
    return _recentre(members=members, centre=centre, alpha=alpha)


execute = recentre
