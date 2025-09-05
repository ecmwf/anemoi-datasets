import json
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from earthkit.data.readers.odb import ODBReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_varno_dict(path: Optional[str] = None) -> Dict:
    """Load varno mapping, return empty dict if not found."""
    try:
        with open(path or "varno.json") as f:
            return json.load(f)
    except:
        return {"data": []}


def get_varno_name(varno: Union[int, str], varno_dict: Dict) -> str:
    """Get varno name or return original if not found."""
    try:
        v = int(varno)
        for entry in varno_dict.get("data", []):
            if v in entry:
                return str(entry[0])
    except:
        pass
    return str(varno)


def rename_cols(cols: List, extra_obs: List[str] = None, varno_path: str = None) -> List[str]:
    """Rename columns: base_name_varno_level"""
    varno_dict = load_varno_dict(varno_path)
    extra_obs = extra_obs or []

    result = []
    for col in cols:
        if isinstance(col, tuple):
            parts = col + ("", "")
            name, varno = parts[:2]
            level = parts[2] if len(parts) > 2 else ""
        else:
            name, varno, level = col, "", ""

        base = name.split("@")[0]
        if base in extra_obs:
            base = f"obsvalue_{base}"

        if varno:
            varno_name = get_varno_name(varno, varno_dict)
            level_str = str(int(level)) if level and not isinstance(level, (list, tuple)) else "0"
            result.append(f"{base}_{varno_name}_{level_str}")
        else:
            result.append(base)

    return result


def process_odb(
    reader: ODBReader,
    index: List[str],
    pivot: List[str],
    values: List[str],
    sort: List[str] = None,
    extra_obs: List[str] = None,
    drop_na: bool = False,
    datetime_cols: tuple = ("date@hdr", "time@hdr"),
    varno_path: str = None,
) -> pd.DataFrame:
    """Process ODB data: convert to pandas, pivot, rename columns."""

    try:
        df = reader.to_pandas()
    except Exception as e:
        logging.error(f"ODB conversion failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Remove duplicates and pivot
    df = df.drop_duplicates(subset=index + pivot, keep="first")
    df = df.pivot(index=index, columns=pivot, values=values)

    # Sort and reset
    if sort and all(c in df.index.names for c in sort):
        df = df.sort_values(by=sort, kind="stable")
    df = df.reset_index()

    # Reorganize columns
    meta = df[index]
    obs = df.drop(columns=index, level=0).sort_index(axis=1)
    df = pd.concat([meta, obs], axis=1)

    if drop_na:
        df = df.dropna()

    # Create datetime if both columns exist
    date_col, time_col = datetime_cols
    if date_col in df.columns and time_col in df.columns:
        try:
            df["times"] = pd.to_datetime(
                df[date_col].astype(int).astype(str) + df[time_col].astype(int).astype(str).str.zfill(6),
                format="%Y%m%d%H%M%S",
            )
            df = df.drop(columns=[date_col, time_col], level=0)
        except:
            logging.warning("Could not create datetime column")

    # Rename columns
    df.columns = rename_cols(df.columns.tolist(), extra_obs, varno_path)

    # Rename lat/lon columns to match expected format
    df = df.rename(columns={"lat": "latitudes", "lon": "longitudes"})

    return df


# Example usage:
# df = process_odb(reader, ["seqno@hdr", "lat@hdr", "lon@hdr"], ["varno@body"], ["obsvalue@body"])
