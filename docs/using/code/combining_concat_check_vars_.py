ds = open_dataset(
    concat=[dataset1, dataset2, ...],
    check_variables_compatibility={
        "ignore_type_of_level": "msl",  # Don't check type of level for the variable "msl"
        "ignore_units": [
            "msl",
            "t2m",
        ],  # Don't check units for the variables "msl" and "t2m"
    },
)
