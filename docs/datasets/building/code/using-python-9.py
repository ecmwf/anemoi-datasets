from anemoi.datasets.recipe import Recipe

r = Recipe()

r.dates = ("2023-01-01T00:00:00", "2023-12-31T18:00:00", "12h")

a = r.grib(path="dir1/*.grib")
b = r.grib(path="dir2/*.grib")
c = r.forcings(param=["cos_latitude", "sin_latitude"], template=a)

r.input = a & b & c

r.dump()
