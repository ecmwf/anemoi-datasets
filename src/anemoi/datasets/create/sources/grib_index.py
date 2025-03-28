# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import sqlite3
from typing import Any
from typing import List
from typing import Optional

import earthkit.data as ekd
import tqdm
from cachetools import LRUCache
from earthkit.data.indexing.fieldlist import FieldArray

from anemoi.datasets.create.sources.grib_support.flavour import RuleBasedFlavour

from .legacy import legacy_source

LOG = logging.getLogger(__name__)


class GribIndex:

    def __init__(
        self, database, *, keys: Optional[List[str]] = None, update: bool = False, overwrite: bool = False
    ) -> None:
        self.database = database
        if overwrite:
            assert update
            if os.path.exists(database):
                os.remove(database)

        if not update:
            if not os.path.exists(database):
                raise FileNotFoundError(f"Database {database} does not exist")

        self.conn = sqlite3.connect(database)
        self.cursor = self.conn.cursor()

        self.update = update
        self.cache = None
        self.keys = keys
        self._columns = None

        if update:
            assert keys is not None
            self._create_tables()
        else:
            assert keys is None
            self.keys = self._all_columns()
            self.cache = LRUCache(maxsize=50)

    def _create_tables(self) -> None:
        assert self.update

        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS paths (
            id INTEGER PRIMARY KEY,
            path TEXT not null
        )
        """
        )

        columns = ("valid_datetime",)
        # We don't use NULL as a default because NULL is considered a different value
        # in UNIQUE INDEX constraints (https://www.sqlite.org/lang_createindex.html)

        self.cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS grib_index (
            _id INTEGER PRIMARY KEY,
            _path_id INTEGER not null,
            _offset INTEGER not null,
            _length INTEGER not null,
            {', '.join(f"{key} TEXT not null default ''" for key in columns)},
            FOREIGN KEY(_path_id) REFERENCES paths(id))
        """
        )  # ,

        self.cursor.execute(
            """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_grib_index_path_offset
        ON grib_index (_path_id, _offset)
        """
        )

        self.cursor.execute(
            f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_grib_index_all_keys
        ON grib_index ({', '.join(columns)})
        """
        )

        for key in columns:
            self.cursor.execute(
                f"""
            CREATE INDEX IF NOT EXISTS idx_grib_index_{key}
            ON grib_index ({key})
            """
            )

        self._commit()

    def _commit(self):
        self.conn.commit()

    def _get_metadata_keys(self) -> List[str]:
        """Retrieve the metadata keys from the database.

        Returns
        -------
        List[str]
            A list of metadata keys stored in the database.
        """
        self.cursor.execute("SELECT key FROM metadata_keys")
        return [row[0] for row in self.cursor.fetchall()]

    def _path_id(self, path: str) -> int:
        """Get the id of a path in the database."""

        self.cursor.execute("SELECT id FROM paths WHERE path = ?", (path,))
        row = self.cursor.fetchone()
        if row is None:
            self.cursor.execute("INSERT INTO paths (path) VALUES (?)", (path,))
            self._commit()
            return self.cursor.lastrowid
        return row[0]

    def _add_grib(self, **kwargs: Any) -> None:
        """Add a grib record to the database."""

        assert self.update

        try:

            self.cursor.execute(
                f"""
            INSERT INTO grib_index ({', '.join(kwargs.keys())})
            VALUES ({', '.join('?' for _ in kwargs)})
            """,
                tuple(kwargs.values()),
            )

        except sqlite3.IntegrityError:
            LOG.error(f"Error adding grib record: {kwargs}")
            LOG.error("Record already exists")
            LOG.info(f"Path: {self._get_path(kwargs['_path_id'])}")
            for n in ("_path_id", "_offset", "_length"):
                kwargs.pop(n)
            self.cursor.execute(
                "SELECT * FROM grib_index WHERE " + " AND ".join(f"{key} = ?" for key in kwargs.keys()),
                tuple(kwargs.values()),
            )
            existing_record = self.cursor.fetchone()
            if existing_record:
                LOG.info(f"Existing record found: {existing_record}")
                LOG.info(f"Path: {self._get_path(existing_record[1])}")
            raise

    def _all_columns(self) -> List[str]:

        if self._columns is not None:
            return self._columns

        self.cursor.execute("PRAGMA table_info(grib_index)")
        columns = {row[1] for row in self.cursor.fetchall()}
        self._columns = [col for col in columns if not col.startswith("_")]
        return self._columns

    def _ensure_columns(self, columns: List[str]) -> None:
        """Add columns to the grib_index table."""

        assert self.update

        existing_columns = self._all_columns()
        new_columns = [column for column in columns if column not in existing_columns]

        if not new_columns:
            return

        self._columns = None

        for column in new_columns:
            self.cursor.execute(f"ALTER TABLE grib_index ADD COLUMN {column} TEXT not null default ''")

        self.cursor.execute("""DROP INDEX IF EXISTS idx_grib_index_all_keys""")
        all_columns = self._all_columns()

        self.cursor.execute(
            f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_grib_index_all_keys
        ON grib_index ({', '.join(all_columns)})
        """
        )

        for key in all_columns:
            self.cursor.execute(
                f"""
            CREATE INDEX IF NOT EXISTS idx_grib_index_{key}
            ON grib_index ({key})
            """
            )

    def add_grib_file(self, path: str) -> None:
        path_id = self._path_id(path)

        for field in tqdm.tqdm(ekd.from_source("file", path), leave=False):

            keys = {k: field.metadata(k, default=None) for k in self.keys}
            keys = {k: v for k, v in keys.items() if v is not None}

            self._ensure_columns(list(keys.keys()))

            self._add_grib(
                _path_id=path_id,
                _offset=field.metadata("offset"),
                _length=field.metadata("totalLength"),
                **keys,
            )

        self._commit()

    def _get_path(self, path_id: int) -> str:
        """Retrieve the path corresponding to a given path_id.

        Parameters
        ----------
        path_id : int
            The ID of the path to retrieve.

        Returns
        -------
        str
            The path corresponding to the given path_id.

        Raises
        ------
        ValueError
            If the path_id does not exist in the database.
        """
        self.cursor.execute("SELECT path FROM paths WHERE id = ?", (path_id,))
        row = self.cursor.fetchone()
        if row is None:
            raise ValueError(f"No path found for path_id {path_id}")
        return row[0]

    def retrieve(self, dates, **kwargs):
        assert not self.update

        dates = [d.isoformat() for d in dates]

        query = """SELECT _path_id, _offset, _length
                   FROM grib_index WHERE valid_datetime IN ({})""".format(
            ", ".join("?" for _ in dates)
        )
        params = dates

        for k, v in kwargs.items():
            if isinstance(v, list):
                query += f" AND {k} IN ({', '.join('?' for _ in v)})"
                params.extend([str(_) for _ in v])
            else:
                query += f" AND {k} = ?"
                params.append(str(v))

        print("SELECT", query)
        print("SELECT", params)

        self.cursor.execute(query, params)
        for path_id, offset, length in self.cursor.fetchall():
            if path_id in self.cache:
                file = self.cache[path_id]
            else:
                path = self._get_path(path_id)
                LOG.info(f"Opening {path}")
                self.cache[path_id] = open(path, "rb")
                file = self.cache[path_id]

            file.seek(offset)
            data = file.read(length)
            yield data


@legacy_source(__file__)
def execute(context, dates, indexdb, flavour=None, **kwargs):
    index = GribIndex(indexdb)
    result = []

    if flavour is not None:
        flavour = RuleBasedFlavour(flavour)

    for grib in index.retrieve(dates, **kwargs):
        field = ekd.from_source("memory", grib)[0]
        if flavour:
            field = flavour.apply(field)
        result.append(field)

    return FieldArray(result)
