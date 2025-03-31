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
from typing import Iterator
from typing import List
from typing import Optional

import earthkit.data as ekd
import tqdm
from anemoi.transform.flavour import RuleBasedFlavour
from cachetools import LRUCache
from earthkit.data.indexing.fieldlist import FieldArray

from .legacy import legacy_source

LOG = logging.getLogger(__name__)

KEYS1 = ("class", "type", "stream", "expver", "levtype")
KEYS2 = ("shortName", "paramId", "level", "step", "number", "date", "time", "valid_datetime", "levelist")

KEYS = KEYS1 + KEYS2


class GribIndex:
    def __init__(
        self,
        database: str,
        *,
        keys: Optional[List[str] | str] = None,
        flavour: Optional[str] = None,
        update: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Initialize the GribIndex object.

        Parameters
        ----------
        database : str
            Path to the SQLite database file.
        keys : Optional[List[str] | str], optional
            List of keys or a string of keys to use for indexing, by default None.
        flavour : Optional[str], optional
            Flavour configuration for mapping fields, by default None.
        update : bool, optional
            Whether to update the database, by default False.
        overwrite : bool, optional
            Whether to overwrite the database if it exists, by default False.
        """
        self.database = database
        if overwrite:
            assert update
            if os.path.exists(database):
                os.remove(database)

        if not update:
            if not os.path.exists(database):
                raise FileNotFoundError(f"Database {database} does not exist")

        if keys is not None:
            if isinstance(keys, str):
                if keys.startswith("+"):
                    keys = set(KEYS) | set(keys[1:].split(","))
                else:
                    keys = set(",".split(keys.split(",")))
                keys = list(keys)

        self.conn = sqlite3.connect(database)
        self.cursor = self.conn.cursor()

        if flavour is not None:
            self.flavour = RuleBasedFlavour(flavour)
        else:
            self.flavour = None

        self.update = update
        self.cache = None
        self.keys = keys
        self._columns = None

        if update:
            if self.keys is None:
                self.keys = KEYS
            LOG.info(f"Using keys: {sorted(self.keys)}")
            self._create_tables()
        else:
            assert keys is None
            self.keys = self._all_columns()
            self.cache = LRUCache(maxsize=50)

        self.warnings = {}
        self.cache = {}

    def _create_tables(self) -> None:
        """Create the necessary tables in the database."""
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

    def _commit(self) -> None:
        """Commit the current transaction to the database."""
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
        """Get the id of a path in the database.

        Parameters
        ----------
        path : str
            The file path to retrieve or insert.

        Returns
        -------
        int
            The ID of the path in the database.
        """
        self.cursor.execute("SELECT id FROM paths WHERE path = ?", (path,))
        row = self.cursor.fetchone()
        if row is None:
            self.cursor.execute("INSERT INTO paths (path) VALUES (?)", (path,))
            self._commit()
            return self.cursor.lastrowid
        return row[0]

    def _add_grib(self, **kwargs: Any) -> None:
        """Add a GRIB record to the database.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs representing the GRIB record fields.
        """
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
        """Retrieve all column names from the grib_index table.

        Returns
        -------
        List[str]
            A list of column names.
        """
        if self._columns is not None:
            return self._columns

        self.cursor.execute("PRAGMA table_info(grib_index)")
        columns = {row[1] for row in self.cursor.fetchall()}
        self._columns = [col for col in columns if not col.startswith("_")]
        return self._columns

    def _ensure_columns(self, columns: List[str]) -> None:
        """Add missing columns to the grib_index table.

        Parameters
        ----------
        columns : List[str]
            List of column names to ensure in the table.
        """
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
        """Add a GRIB file to the database.

        Parameters
        ----------
        path : str
            Path to the GRIB file to add.
        """
        path_id = self._path_id(path)

        fields = ekd.from_source("file", path)
        if self.flavour is not None:
            fields = self.flavour.map(fields)

        for i, field in enumerate(tqdm.tqdm(fields, leave=False)):

            keys = field.metadata(namespace="mars").copy()
            keys.update({k: field.metadata(k, default=None) for k in self.keys})

            keys.setdefault("param", keys.get("shortName", keys.get("paramId")))

            keys = {k: v for k, v in keys.items() if v is not None}

            if keys.get("param") in (0, "unknown"):
                param = (
                    field.metadata("discipline", default=None),
                    field.metadata("parameterCategory", default=None),
                    field.metadata("parameterNumber", default=None),
                )
                if param not in self.warnings:
                    self._unknown(path, field, i, param)
                    self.warnings[param] = True

            self._ensure_columns(list(keys.keys()))

            self._add_grib(
                _path_id=path_id,
                _offset=field.metadata("offset"),
                _length=field.metadata("totalLength"),
                **keys,
            )

        self._commit()

    def _paramdb(self, category: int, discipline: int) -> Optional[dict]:
        """Fetch parameter information from the parameter database.

        Parameters
        ----------
        category : int
            The parameter category.
        discipline : int
            The parameter discipline.

        Returns
        -------
        Optional[dict]
            The parameter information, or None if unavailable.
        """
        if (category, discipline) in self.cache:
            return self.cache[(category, discipline)]

        try:
            import requests

            r = requests.get(
                f"https://codes.ecmwf.int/parameter-database/api/v1/param?category={category}&discipline={discipline}"
            )
            r.raise_for_status()
            self.cache[(category, discipline)] = r.json()
            return self.cache[(category, discipline)]

        except Exception as e:
            LOG.warning(f"Failed to fetch information from parameter database: {e}")

    def _param_grib2_info(self, paramId: int) -> List[dict]:
        """Fetch GRIB2 parameter information for a given parameter ID.

        Parameters
        ----------
        paramId : int
            The parameter ID.

        Returns
        -------
        List[dict]
            A list of GRIB2 parameter information.
        """
        if ("grib2", paramId) in self.cache:
            return self.cache[("grib2", paramId)]

        try:
            import requests

            r = requests.get(f"https://codes.ecmwf.int/parameter-database/api/v1/param/{paramId}/grib2/")
            r.raise_for_status()
            self.cache[("grib2", paramId)] = r.json()
            return self.cache[("grib2", paramId)]

        except Exception as e:
            LOG.warning(f"Failed to fetch information from parameter database: {e}")
        return []

    def _param_id_info(self, paramId: int) -> Optional[dict]:
        """Fetch detailed information for a given parameter ID.

        Parameters
        ----------
        paramId : int
            The parameter ID.

        Returns
        -------
        Optional[dict]
            The parameter information, or None if unavailable.
        """
        if ("info", paramId) in self.cache:
            return self.cache[("info", paramId)]

        try:
            import requests

            r = requests.get(f"https://codes.ecmwf.int/parameter-database/api/v1/param/{paramId}/")
            r.raise_for_status()
            self.cache[("info", paramId)] = r.json()
            return self.cache[("info", paramId)]

        except Exception as e:
            LOG.warning(f"Failed to fetch information from parameter database: {e}")

        return None

    def _param_id_unit(self, unitId: int) -> Optional[dict]:
        """Fetch unit information for a given unit ID.

        Parameters
        ----------
        unitId : int
            The unit ID.

        Returns
        -------
        Optional[dict]
            The unit information, or None if unavailable.
        """
        if ("unit", unitId) in self.cache:
            return self.cache[("unit", unitId)]

        try:
            import requests

            r = requests.get(f"https://codes.ecmwf.int/parameter-database/api/v1/unit/{unitId}/")
            r.raise_for_status()
            self.cache[("unit", unitId)] = r.json()
            return self.cache[("unit", unitId)]

        except Exception as e:
            LOG.warning(f"Failed to fetch information from parameter database: {e}")

        return None

    def _unknown(self, path: str, field: ekd.Field, i: int, param: tuple) -> None:
        """Log information about unknown parameters.

        Parameters
        ----------
        path : str
            Path to the GRIB file.
        field : ekd.Field
            The GRIB field object.
        i : int
            The index of the field in the file.
        param : tuple
            The parameter tuple (discipline, category, parameterNumber).
        """

        def _(s):
            try:
                return int(s)
            except ValueError:
                return s

        LOG.warning(
            f"Unknown param for message {i+1} in {path} at offset {int(field.metadata('offset', default=None))}"
        )
        LOG.warning(
            f"shortName/paramId: {field.metadata('shortName', default=None)}/{field.metadata('paramId', default=None)}"
        )
        name = field.metadata("parameterName", default=None)
        units = field.metadata("parameterUnits", default=None)
        LOG.warning(f"Discipline/category/parameter: {param} ({name}, {units})")
        LOG.warning(f"grib_copy -w count={i+1} {path} tmp.grib")

        info = self._paramdb(discipline=param[0], category=param[1])
        found = set()
        if info is not None:
            for n in tqdm.tqdm(info, desc="Scanning parameter database"):

                for p in self._param_grib2_info(n["id"]):

                    keys = {k["name"]: _(k["value"]) for k in p["keys"]}
                    if keys.get("parameterNumber") == param[2]:
                        found.add(n["id"])

        for n in found:
            info = self._param_id_info(n)
            if "unit_id" in info:
                info["unit_id"] = self._param_id_unit(info["unit_id"])["name"]

            LOG.info("%s", f"Possible match: {n}")
            LOG.info("%s", f"     Name:        {info.get('name')}")
            LOG.info("%s", f"     Short name:  {info.get('shortname')}")
            LOG.info("%s", f"     Units:       {info.get('unit_id')}")
            LOG.info("%s", f"     Description: {info.get('description')}")
            LOG.info("")

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

    def retrieve(self, dates: List[Any], **kwargs: Any) -> Iterator[Any]:
        """Retrieve GRIB data from the database.

        Parameters
        ----------
        dates : List[Any]
            List of dates to retrieve data for.
        **kwargs : Any
            Additional filtering criteria.

        Returns
        ------
        Iterator[Any]
            The GRIB data matching the criteria.
        """
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
def execute(
    context: Any,
    dates: List[Any],
    indexdb: str,
    flavour: Optional[str] = None,
    **kwargs: Any,
) -> FieldArray:
    """Execute the GRIB data retrieval process.

    Parameters
    ----------
    context : Any
        The execution context.
    dates : List[Any]
        List of dates to retrieve data for.
    indexdb : str
        Path to the GRIB index database.
    flavour : Optional[str], optional
        Flavour configuration for mapping fields, by default None.
    **kwargs : Any
        Additional filtering criteria.

    Returns
    -------
    FieldArray
        An array of retrieved GRIB fields.
    """
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
