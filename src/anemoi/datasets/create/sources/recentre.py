# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from anemoi.datasets.compute.recentre import recentre as _recentre

from .legacy import legacy_source
from .mars import mars


def to_list(x: Union[list, tuple, str]) -> List:
    """Converts the input to a list. If the input is a string, it splits it by '/'.

    Parameters
    ----------
    x : Union[list, tuple, str]
        The input to convert.

    Returns
    -------
    list
        The converted list.
    """
    if isinstance(x, (list, tuple)):
        return x
    if isinstance(x, str):
        return x.split("/")
    return [x]


def normalise_number(number: Union[list, tuple, str]) -> List[int]:
    """Normalises the input number to a list of integers.

    Parameters
    ----------
    number : Union[list, tuple, str]
        The number to normalise.

    Returns
    -------
    list
        The normalised list of integers.
    """
    number = to_list(number)

    if len(number) > 4 and (number[1] == "to" and number[3] == "by"):
        return list(range(int(number[0]), int(number[2]) + 1, int(number[4])))

    if len(number) > 2 and number[1] == "to":
        return list(range(int(number[0]), int(number[2]) + 1))

    return number


def normalise_request(request: Dict) -> Dict:
    """Normalises the request dictionary by converting certain fields to lists.

    Parameters
    ----------
    request : dict
        The request dictionary to normalise.

    Returns
    -------
    dict
        The normalised request dictionary.
    """
    request = deepcopy(request)
    if "number" in request:
        request["number"] = normalise_number(request["number"])
    if "time" in request:
        request["time"] = to_list(request["time"])
    request["param"] = to_list(request["param"])
    return request


def load_if_needed(context: Any, dates: Any, dict_or_dataset: Union[Dict, Any]) -> Any:
    """Loads the dataset if the input is a dictionary, otherwise returns the input.

    Parameters
    ----------
    context : Any
        The context for loading the dataset.
    dates : Any
        The dates for loading the dataset.
    dict_or_dataset : Union[dict, Any]
        The input dictionary or dataset.

    Returns
    -------
    Any
        The loaded dataset or the original input.
    """
    if isinstance(dict_or_dataset, dict):
        dict_or_dataset = normalise_request(dict_or_dataset)
        dict_or_dataset = mars(context, dates, dict_or_dataset)
    return dict_or_dataset


@legacy_source(__file__)
def recentre(
    context: Any,
    dates: Any,
    members: Union[Dict, Any],
    centre: Union[Dict, Any],
    alpha: float = 1.0,
    remapping: Dict = {},
    patches: Dict = {},
) -> Any:
    """Recentres the members dataset using the centre dataset.

    Parameters
    ----------
    context : Any
        The context for recentering.
    dates : Any
        The dates for recentering.
    members : Union[dict, Any]
        The members dataset or request dictionary.
    centre : Union[dict, Any]
        The centre dataset or request dictionary.
    alpha : float, optional
        The alpha value for recentering. Defaults to 1.0.
    remapping : dict, optional
        The remapping dictionary. Defaults to {}.
    patches : dict, optional
        The patches dictionary. Defaults to {}.

    Returns
    -------
    Any
        The recentred dataset.
    """
    members = load_if_needed(context, dates, members)
    centre = load_if_needed(context, dates, centre)
    return _recentre(members=members, centre=centre, alpha=alpha)


execute = recentre
