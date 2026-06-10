# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import sys
from typing import Any

import yaml

from . import Command

LOG = logging.getLogger(__name__)


def _json_default(obj: Any) -> str:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    return str(obj)


def _date_repr(d: Any) -> Any:
    if isinstance(d, tuple):
        return [_date_repr(x) for x in d]
    if isinstance(d, (datetime.datetime, datetime.date)):
        return d.isoformat()
    return str(d)


def _serialise_group(group: Any) -> dict:
    dates = [_date_repr(d) for d in group]
    return {
        "n_dates": len(dates),
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "dates": dates,
    }


class AnalyseRecipe(Command):
    """Analyse a dataset recipe and output a JSON report on stdout.

    The recipe is loaded and validated exactly as it would be when
    creating a dataset (through pydantic). This command is intended to
    be called by external tools (e.g. prepml) so that the analysis is
    performed by the same version of anemoi-datasets that will build
    the dataset, instead of importing anemoi.datasets in the caller's
    environment.
    """

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        command_parser.add_argument("path", metavar="RECIPE", help="Path to the recipe (YAML).")
        command_parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Include the normalised recipes and the full list of groups of dates.",
        )

    def run(self, args: Any) -> None:
        """Run the command.

        Parameters
        ----------
        args : Any
            The command arguments.
        """
        from pydantic import ValidationError

        from anemoi.datasets import __version__
        from anemoi.datasets.create.recipe import Recipe

        with open(args.path) as f:
            raw = yaml.safe_load(f)

        result: dict = {
            "anemoi_datasets_version": __version__,
            "recipe": raw,
        }

        try:
            recipe = Recipe(**raw)
        except ValidationError as e:
            result["valid"] = False
            result["errors"] = [
                {
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                }
                for error in e.errors()
            ]
            print(json.dumps(result, indent=2, default=_json_default))
            sys.exit(1)

        result["valid"] = True

        layout = getattr(recipe.output, "layout", None)

        # Recipes coming from prepml carry an extra 'name' key; when present,
        # check it against the dataset naming conventions of the recipe layout.
        name = raw.get("name")
        if name is not None:
            from anemoi.datasets.create.naming import check_dataset_name

            result["name"] = name
            result["name_errors"] = list(check_dataset_name(name, layout=layout))

        dump = json.loads(recipe.model_dump_json())

        if args.verbose:
            result["recipe_without_defaults"] = recipe.only_non_defaults(dump)
            result["recipe_with_defaults"] = dump

        groups = recipe.make_groups()

        result["layout"] = layout
        result["group_by"] = recipe.build.group_by
        result["n_groups"] = len(groups)

        group_list = [_serialise_group(g) for g in groups]
        result["n_dates"] = sum(g["n_dates"] for g in group_list)
        result["group_lengths"] = [g["n_dates"] for g in group_list]

        result["first_date"] = groups.first_date().isoformat()
        result["last_date"] = groups.last_date().isoformat()

        frequency = getattr(groups.provider, "frequency", None)
        if frequency is not None:
            from anemoi.utils.dates import frequency_to_string

            frequency = frequency_to_string(frequency)
        result["frequency"] = frequency

        if args.verbose:
            result["groups"] = group_list

        print(json.dumps(result, indent=2, default=_json_default))


command = AnalyseRecipe
