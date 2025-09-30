def task_factory(name: str, fields: bool, trace: str | None = None, **kwargs):
    if fields:
        from anemoi.datasets.create.fields.tasks import task_factory as fields_task_factory

        return fields_task_factory(name, trace=trace, **kwargs)
    else:
        from anemoi.datasets.create.observations.tasks import task_factory as observations_task_factory

        return observations_task_factory(name, trace=trace, **kwargs)
