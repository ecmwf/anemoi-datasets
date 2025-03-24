# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.meteo import constants
from earthkit.meteo import thermo
from numpy.typing import NDArray

from .legacy import legacy_filter


# Alternative proposed by Baudouin Raoult
class AutoDict(dict):
    """A dictionary that automatically creates nested dictionaries for missing keys."""

    def __missing__(self, key: Any) -> Any:
        """Handle missing keys by creating nested dictionaries.

        Parameters
        ----------
        key : Any
            The missing key.

        Returns
        -------
        Any
            A new nested dictionary.
        """
        value = self[key] = type(self)()
        return value


def model_level_pressure(
    A: NDArray[Any], B: NDArray[Any], surface_pressure: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates:
     - pressure at the model full- and half-levels
     - delta: depth of log(pressure) at full levels
     - alpha: alpha term #TODO: more descriptive information.

    Parameters
    ----------
    A : ndarray
        A-coefficients defining the model levels
    B : ndarray
        B-coefficients defining the model levels
    surface_pressure : number or ndarray
        surface pressure (Pa)

    Returns
    -------
    ndarray
        pressure at model full-levels
    ndarray
        pressure at model half-levels
    ndarray
        delta at full-levels
    ndarray
        alpha at full levels
    """

    # constants
    PRESSURE_TOA = 0.1  # safety when highest pressure level = 0.0

    # make the calculation agnostic to the number of dimensions
    ndim = surface_pressure.ndim
    new_shape_half = (A.shape[0],) + (1,) * ndim
    A_reshaped = A.reshape(new_shape_half)
    B_reshaped = B.reshape(new_shape_half)

    # calculate pressure on model half-levels
    p_half_level = A_reshaped + B_reshaped * surface_pressure[np.newaxis, ...]

    # calculate delta
    new_shape_full = (A.shape[0] - 1,) + surface_pressure.shape
    delta = np.zeros(new_shape_full)
    delta[1:, ...] = np.log(p_half_level[2:, ...] / p_half_level[1:-1, ...])

    # pressure at highest half level<= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        delta[0, ...] = np.log(p_half_level[1, ...] / PRESSURE_TOA)
    # pressure at highest half level > 0.1
    else:
        delta[0, ...] = np.log(p_half_level[1, ...] / p_half_level[0, ...])

    # calculate alpha
    alpha = np.zeros(new_shape_full)

    alpha[1:, ...] = 1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]

    # pressure at highest half level <= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        alpha[0, ...] = 1.0  # ARPEGE choice, ECMWF IFS uses log(2)
    # pressure at highest half level > 0.1
    else:
        alpha[0, ...] = 1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]

    # calculate pressure on model full levels
    # TODO: is there a faster way to calculate the averages?
    # TODO: introduce option to calculate full levels in more complicated way
    p_full_level = np.apply_along_axis(lambda m: np.convolve(m, np.ones(2) / 2, mode="valid"), axis=0, arr=p_half_level)

    return p_full_level, p_half_level, delta, alpha


def calc_specific_gas_constant(q: Union[float, np.ndarray]) -> Union[float, NDArray[Any]]:
    """Calculates the specific gas constant of moist air
    (specific content of cloud particles and hydrometeors are neglected).

    Parameters
    ----------
    q : number or ndarray
        specific humidity

    Returns
    -------
    number or ndarray
        specific gas constant of moist air
    """

    R = constants.Rd + (constants.Rv - constants.Rd) * q
    return R


def relative_geopotential_thickness(alpha: NDArray[Any], q: NDArray[Any], T: NDArray[Any]) -> NDArray[Any]:
    """Calculates the geopotential thickness w.r.t the surface on model full-levels.

    Parameters
    ----------
    alpha : ndarray
        alpha term of pressure calculations
    q : ndarray
        specific humidity (in kg/kg) on model full-levels
    T : ndarray
        temperature (in Kelvin) on model full-levels

    Returns
    -------
    ndarray
        geopotential thickness of model full-levels w.r.t. the surface
    """

    R = calc_specific_gas_constant(q)
    dphi = np.cumsum(np.flip(alpha * R * T, axis=0), axis=0)
    dphi = np.flip(dphi, axis=0)

    return dphi


def pressure_at_height_level(
    height: float, q: NDArray[Any], T: NDArray[Any], sp: NDArray[Any], A: NDArray[Any], B: NDArray[Any]
) -> Union[float, NDArray[Any]]:
    """Calculates the pressure at a height level given in meters above surface.
    This is done by finding the model level above and below the specified height
    and interpolating the pressure.

    Parameters
    ----------
    height : number
        height (in meters) above the surface for which the pressure is wanted
    q : ndarray
        specific humidity (kg/kg) at model full-levels
    T : ndarray
        temperature (K) at model full-levels
    sp : ndarray
        surface pressure (Pa)
    A : ndarray
        A-coefficients defining the model levels
    B : ndarray
        B-coefficients defining the model levels

    Returns
    -------
    number or ndarray
        pressure (Pa) at the given height level
    """

    # geopotential thickness of the height level
    tdphi = height * constants.g

    # pressure(-related) variables
    p_full, p_half, _, alpha = model_level_pressure(A, B, sp)

    # relative geopot. thickness of full levels
    dphi = relative_geopotential_thickness(alpha, q, T)

    # find the model full level right above the height level
    i_phi = (tdphi > dphi).sum(0)

    # initialize the output array
    p_height = np.zeros_like(i_phi, dtype=np.float64)

    # define mask: requested height is below the lowest model full-level
    mask = i_phi == 0

    # CASE 1: requested height is below the lowest model full-level
    # --> interpolation between surface pressure and lowest model full-level
    p_height[mask] = (p_half[-1, ...] + tdphi / dphi[-1, ...] * (p_full[-1, ...] - p_half[-1, ...]))[mask]

    # CASE 2: requested height is above the lowest model full-level
    # --> interpolation between between model full-level above and below

    # define some indices for masking and readability
    i_lev = alpha.shape[0] - i_phi - 1  # convert phi index to model level index
    indices = np.indices(i_lev.shape)
    masked_indices = tuple(dim[~mask] for dim in indices)
    above = (i_lev[~mask],) + masked_indices
    below = (i_lev[~mask] + 1,) + masked_indices

    dphi_above = dphi[above]
    dphi_below = dphi[below]

    factor = (tdphi - dphi_above) / (dphi_below - dphi_above)
    p_height[~mask] = p_full[above] + factor * (p_full[below] - p_full[above])

    return p_height


@legacy_filter(__file__)
def execute(
    context: Any,
    input: List[Any],
    height: float,
    t: str,
    q: str,
    sp: str,
    new_name: str = "2r",
    **kwargs: Dict[str, Any],
) -> ekd.FieldList:
    """Convert the single (height) level specific humidity to relative humidity.

    Parameters
    ----------
    context : Any
        The context for the execution.
    input : list of Any
        The input data.
    height : float
        The height level in meters.
    t : str
        The temperature parameter name.
    q : str
        The specific humidity parameter name.
    sp : str
        The surface pressure parameter name.
    new_name : str, optional
        The new name for the relative humidity parameter, by default "2r".
    **kwargs : dict
        Additional keyword arguments.
        t_ml : str, optional
            The temperature parameter name for model levels, by default "t".
        q_ml : str, optional
            The specific humidity parameter name for model levels, by default "q".
        A : list of float
            A-coefficients defining the model levels.
        B : list of float
            B-coefficients defining the model levels.
        keep_q : bool, optional
            Whether to keep the specific humidity field in the result, by default False.

    Returns
    -------
    ekd.FieldList
        The resulting field array with relative humidity.
    """
    result = []

    MANDATORY_KEYS = ["A", "B"]
    OPTIONAL_KEYS = ["t_ml", "q_ml"]
    MISSING_KEYS = []
    DEFAULTS = dict(t_ml="t", q_ml="q")

    for key in OPTIONAL_KEYS:
        if key not in kwargs:
            print(f"key {key} not found in yaml-file, using default key: {DEFAULTS[key]}")
            kwargs[key] = DEFAULTS[key]

    for key in MANDATORY_KEYS:
        if key not in kwargs:
            MISSING_KEYS.append(key)

    if MISSING_KEYS:
        raise KeyError(f"Following keys are missing: {', '.join(MISSING_KEYS)}")

    single_level_params = (t, q, sp)
    model_level_params = (kwargs["t_ml"], kwargs["q_ml"])

    needed_fields = AutoDict()

    # Gather all necessary fields
    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        # check single level parameters
        if param in single_level_params:
            levtype = key.pop("levtype")
            key = tuple(sorted(key.items()))

            if param in needed_fields[key][levtype]:
                raise ValueError(f"Duplicate single level field {param} for {key}")

            needed_fields[key][levtype][param] = f
            if param == q:
                if kwargs.get("keep_q", False):
                    result.append(f)
            else:
                result.append(f)

        # check model level parameters
        elif param in model_level_params:
            levtype = key.pop("levtype")
            levelist = key.pop("levelist")
            key = tuple(sorted(key.items()))

            if param in needed_fields[key][levtype][levelist]:
                raise ValueError(f"Duplicate model level field {param} for {key} at level {levelist}")

            needed_fields[key][levtype][levelist][param] = f

        # all other parameters
        else:
            result.append(f)

    for _, values in needed_fields.items():
        # some checks
        if len(values["sfc"]) != 3:
            raise ValueError("Missing surface fields")

        q_sl = values["sfc"][q].to_numpy(flatten=True)
        t_sl = values["sfc"][t].to_numpy(flatten=True)
        sp_sl = values["sfc"][sp].to_numpy(flatten=True)

        nlevels = len(kwargs["A"]) - 1
        if len(values["ml"]) != nlevels:
            raise ValueError("Missing model levels")

        for key in values["ml"].keys():
            if len(values["ml"][key]) != 2:
                raise ValueError(f"Missing field on level {key}")

        # create 3D arrays for upper air fields
        levels = list(values["ml"].keys())
        levels.sort()
        t_ml = []
        q_ml = []
        for level in levels:
            t_ml.append(values["ml"][level][kwargs["t_ml"]].to_numpy(flatten=True))
            q_ml.append(values["ml"][level][kwargs["q_ml"]].to_numpy(flatten=True))

        t_ml = np.stack(t_ml)
        q_ml = np.stack(q_ml)

        # actual conversion from qv --> rh
        # FIXME:
        # For now We need to go from qv --> td --> rh to take into account
        # the mixed / ice phase when T ~ 0C / T < 0C
        # See https://github.com/ecmwf/earthkit-meteo/issues/15
        p_sl = pressure_at_height_level(height, q_ml, t_ml, sp_sl, np.array(kwargs["A"]), np.array(kwargs["B"]))
        td_sl = thermo.dewpoint_from_specific_humidity(q=q_sl, p=p_sl)
        rh_sl = thermo.relative_humidity_from_dewpoint(t=t_sl, td=td_sl)

        result.append(new_field_from_numpy(values["sfc"][q], rh_sl, param=new_name))

    return new_fieldlist_from_list(result)


def test() -> None:
    """Test the conversion from specific humidity to relative humidity.

    This function fetches data from a source, performs the conversion, and prints
    the mean, median, and maximum differences in dewpoint temperature.

    Returns
    -------
    None
    """
    from earthkit.data import from_source
    from earthkit.data.readers.grib.index import GribFieldList

    # IFS forecasts have both specific humidity and dewpoint
    sl = from_source(
        "mars",
        {
            "date": "2022-01-01",
            "class": "od",
            "expver": "1",
            "stream": "oper",
            "levtype": "sfc",
            "param": "96.174/134.128/167.128/168.128",
            "time": "00:00:00",
            "type": "fc",
            "step": "2",
            "grid": "O640",
        },
    )

    ml = from_source(
        "mars",
        {
            "date": "2022-01-01",
            "class": "od",
            "expver": "1",
            "stream": "oper",
            "levtype": "ml",
            "levelist": "130/131/132/133/134/135/136/137",
            "param": "130/133",
            "time": "00:00:00",
            "type": "fc",
            "step": "2",
            "grid": "O640",
        },
    )
    source = GribFieldList.merge([sl, ml])

    # IFS A and B coeffients for level 137 - 129
    kwargs = {
        "A": [424.414063, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.0, 0.0],
        "B": [0.969513, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000],
    }
    source = execute(None, source, 2, "2t", "2sh", "sp", "2r", **kwargs)

    temperature = source[2].to_numpy(flatten=True)
    dewpoint = source[3].to_numpy(flatten=True)
    relhum = source[4].to_numpy()
    newdew = thermo.dewpoint_from_relative_humidity(temperature, relhum)

    print(f"Mean difference in dewpoint temperature: {np.abs(newdew - dewpoint).mean():02f} degC")
    print(f"Median difference in dewpoint temperature: {np.median(np.abs(newdew - dewpoint)):02f} degC")
    print(f"Maximum difference in dewpoint temperature: {np.abs(newdew - dewpoint).max():02f} degC")

    # source.save("source.grib")


if __name__ == "__main__":
    test()
