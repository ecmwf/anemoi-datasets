import datetime

from anemoi.datasets.recipe import Recipe

r = Recipe()

r.description = """
This is a complex example of a dataset recipe written in Python.
It uses data from two different ECMWF research experiments for atmospheric and wave data,
from ECMWF's MARS archive. For the atmospheric data, it combines data from two
12-hourly data streams (oper and lwda) to create a dataset with a  6-hourly frequency.
"""

r.name = "aifs-rd-an-oper-ioku-mars-n320-2024-2024-6h-v1"
r.licence = "CC-BY-4.0"
r.attribution = "ECMWF"

start_date = datetime.datetime(2024, 5, 2, 0, 0)
end_date = datetime.datetime(2024, 9, 8, 18, 0)

r.dates = {
    "start": start_date,
    "end": end_date,
    "frequency": "6h",
}

r.build = {"use_grib_paramid": True}
r.statistics = {"allow_nans": True}


grid = "n320"

ioku = {
    "class": "rd",
    "grid": grid,
    "expver": "ioku",
}

ikdi = {
    "class": "rd",
    "grid": grid,
    "expver": "ikdi",
}

accumulations_stream = {
    "oper": "lwda",
    "lwda": "oper",
}


def accumulations(stream):
    return r.accumulations(
        levtype="sfc",
        param=["cp", "tp", "sf", "strd", "ssrd"],
        stream=accumulations_stream[stream],
        **ioku,
    )


def pressure_levels(stream):
    return r.mars(
        stream=stream,
        level=[
            1,
            10,
            30,
            50,
            70,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            850,
            925,
            1000,
        ],
        levtype="pl",
        param=["t", "u", "v", "w", "z"],
        **ioku,
    )


def pressure_levels_q(stream):
    return r.mars(
        levtype="pl",
        param=["q"],
        level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        stream=stream,
        **ioku,
    )


def sfc_fields(stream):
    return r.mars(
        levtype="sfc",
        param=[
            "10u",
            "10v",
            "2d",
            "2t",
            "lsm",
            "msl",
            "sdor",
            "skt",
            "slor",
            "tcw",
            "z",
            # Land parameters below
            "stl1",
            "stl2",
            "tcc",
            "mcc",
            "hcc",
            "lcc",
            "100u",
            "100v",
        ],
        stream=stream,
        **ioku,
    )


def surface_pressure(stream):
    return (
        r.mars(
            levtype="ml",
            levelist=1,
            param="lnsp",
            stream=stream,
            **ioku,
        )
        | r.lnsp_to_sp()
    )


def apply_mask():
    return r.apply_mask(
        path="/data/climate.v015/319_3/lsm.grib",
        mask_value=0,
    )


def land_params(stream):
    soil_params = r.mars(
        levtype="sfc",
        param=["swvl1", "swvl2", "sd"],
        stream=stream,
        **ioku,
    )

    snow_cover = (
        r.mars(
            levtype="sfc",
            param=["sd", "rsn"],
            stream=stream,
            **ioku,
        )
        | r.snow_cover()
    )

    run_off = r.accumulations(
        levtype="sfc",
        param=["ro"],
        stream=accumulations_stream[stream],
        **ioku,
    )

    return (soil_params & snow_cover & run_off) | apply_mask()


def constants(template):
    return r.constants(
        param=[
            "cos_latitude",
            "cos_longitude",
            "sin_latitude",
            "sin_longitude",
            "cos_julian_day",
            "cos_local_time",
            "sin_julian_day",
            "sin_local_time",
            "insolation",
        ],
        template=template,
    )


def wave_data():
    return (
        r.mars(
            param=[
                "swh",
                "cdww",
                "mwp",
                "mwd",
                "wmb",
                "h1012",
                "h1214",
                "h1417",
                "h1721",
                "h2125",
                "h2530",
            ],
            stream="wave",
            **ikdi,
        )
        | r.cos_sin_mean_wave_direction()
    )


def atmos_data(stream):
    return (
        (a := sfc_fields(stream))
        & surface_pressure(stream)
        & pressure_levels(stream)
        & pressure_levels_q(stream)
        & accumulations(stream)
        & land_params(stream)
        & constants(template=a)
    )


def dates(hour):
    s = start_date.replace(hour=hour)
    e = end_date.replace(hour=hour + 12)
    while s > start_date:
        s -= datetime.timedelta(hours=24)
    while e < end_date:
        e += datetime.timedelta(hours=24)
    return (s, e, "12h")


def input_data():
    return r.concat(
        {
            dates(0): atmos_data("oper"),
            dates(6): atmos_data("lwda"),
        }
    )


r.input = input_data() & wave_data()

r.dump()
