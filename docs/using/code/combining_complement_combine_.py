from anemoi.datasets import open_dataset

ds = open_dataset(
    combine=[
        {"dataset": dataset1, "option1": value1, "option2": ...},
        {"dataset": dataset2, "option3": value3, "option4": ...},
    ]
)
