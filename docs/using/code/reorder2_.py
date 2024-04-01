ds = open_dataset(
    dataset,
    reorder={
        "2t": 0,
        "msl": 1,
        "sp": 2,
        "10u": 3,
        "10v": 4,
    },
)
