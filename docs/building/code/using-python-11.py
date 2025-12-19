from anemoi.datasets.recipe import Recipe

r = Recipe()

r.dates = ("2023-01-01T00:00:00", "2023-12-31T18:00:00", "12h")

r.input = r.concat(
    {
        ("2023-01-01T00:00:00", "2023-06-30T18:00:00", "12h"): r.grib(path="gribs/*.grib"),
        ("2023-07-01T00:00:00", "2023-12-31T18:00:00", "12h"): r.netcdf(path="ncdfs/*.nc"),
    }
)

r.dump()
