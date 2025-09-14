from anemoi.datasets.recipe import Recipe

r = Recipe()

r.input = r.grib("input_data.grib", param=["2t", "msl"])

r.dump()
