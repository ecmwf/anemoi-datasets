from datetime import datetime

from anemoi.datasets.recipe import Recipe

r = Recipe()

# As a tuple (start, end, frequency)
r.dates = ("2023-01-01T00:00:00", "2023-12-31T18:00:00", "12h")

# As a dictionary
r.dates = {
    "start": "2023-01-01T00:00:00",
    "end": "2023-12-31T18:00:00",
    "frequency": "12h",
}

# You can also provide datetime objects

r.dates = {
    "start": datetime(2023, 1, 1, 0, 0),
    "end": datetime(2023, 12, 31, 18, 0),
    "frequency": "12h",
}
