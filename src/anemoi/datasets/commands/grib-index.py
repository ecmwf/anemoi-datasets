# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import fnmatch
import os
import sqlite3
from typing import Any
from typing import List

import earthkit.data as ekd
import tqdm

from . import Command

KEYS1 = ("class", "type", "stream", "expver", "levtype")
KEYS2 = ("shortName", "paramId", "level", "step", "number", "date", "time", "valid_datetime", "levelist")

KEYS = KEYS1 + KEYS2


def open_database(path: str, metadata_keys: List[str]) -> sqlite3.Connection:
    """Open a connection to a sqlite3 database.

    Parameters
    ----------
    path : str
        The path to the database.
    metadata_keys : List[str]
        The list of metadata keys to be used in the database.

    Returns
    -------
    sqlite3.Connection
        The connection to the database.
    """
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS paths (
        id INTEGER PRIMARY KEY,
        path TEXT not null
    )
    """
    )

    cursor.execute(
        f"""
    CREATE TABLE IF NOT EXISTS grib_index (
        id INTEGER PRIMARY KEY,
        path_id INTEGER not null,
        offset INTEGER not null,
        length INTEGER not null,
        {', '.join(f'{key} TEXT' for key in metadata_keys)},
        FOREIGN KEY(path_id) REFERENCES paths(id))
    """
    )

    cursor.execute(
        """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_grib_index_path_offset
    ON grib_index (path_id, offset)
    """
    )

    cursor.execute(
        f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_grib_index_all_keys
    ON grib_index ({', '.join(metadata_keys)})
    """
    )

    for key in metadata_keys:
        cursor.execute(
            f"""
        CREATE INDEX IF NOT EXISTS idx_grib_index_{key}
        ON grib_index ({key})
        """
        )

    conn.commit()

    return conn


def path_id(conn: sqlite3.Connection, path: str) -> int:
    """Get the id of a path in the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    path : str
        The path to look up.

    Returns
    -------
    int
        The id of the path.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM paths WHERE path = ?", (path,))
    row = cursor.fetchone()
    if row is None:
        cursor.execute("INSERT INTO paths (path) VALUES (?)", (path,))
        conn.commit()
        return cursor.lastrowid
    return row[0]


def add_grib(conn: sqlite3.Connection, commit: bool = True, **kwargs: Any) -> None:
    """Add a grib record to the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    commit : bool, optional
        Whether to commit the transaction immediately, by default True.
    **kwargs : Any
        The metadata fields and their values to be added to the database.
    """
    cursor = conn.cursor()

    cursor.execute(
        f"""
    INSERT INTO grib_index ({', '.join(kwargs.keys())})
    VALUES ({', '.join('?' for _ in kwargs)})
    """,
        tuple(kwargs.values()),
    )

    if commit:
        conn.commit()


class GribIndex(Command):
    internal = True
    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser to which arguments are added.
        """
        command_parser.add_argument(
            "--index",
            help="Create an index file",
            required=True,
        )

        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Over write the index file",
        )

        command_parser.add_argument(
            "--match",
            help="Give a glob pattern to match files (default: *.grib)",
            default="*.grib",
        )

        command_parser.add_argument("paths", nargs="+", help="Paths to scan")

    def run(self, args: Any) -> None:
        """Execute the scan command.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """

        def match(path: str) -> bool:
            """Check if a path matches the given pattern.

            Parameters
            ----------
            path : str
                The path to check.

            Returns
            -------
            bool
                True if the path matches, False otherwise.
            """
            return fnmatch.fnmatch(path, args.match)

        if args.overwrite:
            if os.path.exists(args.index):
                os.remove(args.index)

        # Remove namespace if present
        conn = open_database(args.index, [k.split(".")[-1] for k in KEYS])

        paths = []
        for path in args.paths:
            if os.path.isfile(path):
                paths.append(path)
            else:
                for root, _, files in os.walk(path):
                    for file in files:
                        full = os.path.join(root, file)
                        paths.append(full)

        for path in tqdm.tqdm(paths, leave=False):
            if not match(path):
                continue

            path_id_ = path_id(conn, path)

            for field in tqdm.tqdm(ekd.from_source("file", path), leave=False):

                keys = {k.split(".")[-1]: field.metadata(k, default=None) for k in KEYS}

                add_grib(
                    conn,
                    commit=False,
                    path_id=path_id_,
                    offset=field.metadata("offset"),
                    length=field.metadata("totalLength"),
                    **keys,
                )

        conn.commit()


command = GribIndex
