ds = open_dataset(
    concat=[dataset1, dataset2, ...],
    check_variables_compatibility={
        "ignore_units": True,  # Don't check units
        "ignore_time_processing": True,  # Don't check time processing (e.g. whether the data are instantaneous or accumulated)
        "ignore_processing_period": True,  # Don't check time processing period (e.g. whether the data are 3-hourly or 6-hourly accumulations)
        "ignore_type_of_level": True,  # Don't check type of level (e.g. whether the data are on pressure levels or model levels)
    },
)
