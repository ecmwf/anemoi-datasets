# Scale and offset can be passed as a dictionnary...

ds = open_dataset(
    dataset,
    rescale={"2t": {"scale": 1.0, "offset": -273.15}},
)

# ... a tuple of floating points ....

ds = open_dataset(
    dataset,
    rescale={"2t": (1.0, -273.15)},
)

# ... or a tuple of strings representing units.

ds = open_dataset(
    dataset,
    rescale={"2t": ("K", "degC")},
)

# Several variables can be rescaled at once.

ds = open_dataset(
    dataset,
    rescale={
        "2t": ("K", "degC"),
        "tp": ("m", "mm"),
    },
)
