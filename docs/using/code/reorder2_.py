ds = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
    reorder={"2t": 0, "msl": 1, "sp": 2, "10u": 3, "10v": 4},
)
