ds = open_dataset(
    dataset,
    start=1979,
    end=2020,
    window="(-6h,0]",
    frequency="6h",
)
