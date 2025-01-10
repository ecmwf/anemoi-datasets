open_dataset(
    cutout=[
        {
            "complement": lam_dataset,
            "source": global_dataset,
            "interpolate": "nearest",
        },
        {
            "dataset": global_dataset,
        },
    ]
)
