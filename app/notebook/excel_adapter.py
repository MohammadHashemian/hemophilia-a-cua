"""Thin pandas+openpyxl adapter for Excel I/O.

Polars does not support reading or writing Excel files. To keep the
``xlsx`` inputs and outputs of the preprocessing notebook working
without dragging pandas into the rest of the data pipeline, this
module exposes ``read_excel`` and ``write_excel`` that take/return
polars ``DataFrame`` instances.

Polars must be installed; pandas and openpyxl are the only third-party
deps here. This is the **only** place in the project where pandas is
used at I/O boundaries — every other module is polars-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

__all__ = ["read_excel", "write_excel", "write_excel_multi"]


def read_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    **pandas_kwargs: Any,
) -> pl.DataFrame:
    """Read an Excel file and return a polars ``DataFrame``.

    Parameters
    ----------
    path:
        Path to the ``.xlsx`` (or ``.xls``) file.
    sheet_name:
        Sheet to read. Either the sheet name (str) or its 0-based index
        (int). Forwarded to ``pandas.read_excel``.
    **pandas_kwargs:
        Extra keyword arguments forwarded to ``pandas.read_excel`` —
        e.g. ``header``, ``usecols``, ``dtype``, ``nrows``.
    """
    # Local import so the rest of the package can stay pandas-free at
    # import time. pandas is only needed when the caller actually asks
    # to read or write Excel.
    import pandas as pd

    pdf = pd.read_excel(path, sheet_name=sheet_name, **pandas_kwargs)
    return pl.from_pandas(pdf)


def write_excel(
    df: pl.DataFrame,
    path: str | Path,
    sheet_name: str = "Sheet1",
    index: bool = False,
    **pandas_kwargs: Any,
) -> None:
    """Write a polars ``DataFrame`` to an Excel file.

    Parameters
    ----------
    df:
        Polars DataFrame to serialize.
    path:
        Destination ``.xlsx`` path. Parent directories are created.
    sheet_name:
        Sheet name. Defaults to ``"Sheet1"``.
    index:
        If ``True``, the polars index column (set with
        ``df.with_row_index()``) is written as the first column. This
        matches the default ``index=False`` behaviour of pandas
        ``to_excel``.
    **pandas_kwargs:
        Extra keyword arguments forwarded to
        ``pandas.DataFrame.to_excel`` — e.g. ``na_rep``, ``float_format``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf = df.to_pandas()
    pdf.to_excel(path, sheet_name=sheet_name, index=index, **pandas_kwargs)


def write_excel_multi(
    frames: dict[str, pl.DataFrame],
    path: str | Path,
    index: bool = False,
    **pandas_kwargs: Any,
) -> None:
    r"""Write multiple polars DataFrames to a single Excel file,
    one per sheet.

    Parameters
    ----------
    frames:
        Mapping of sheet name -> polars DataFrame. The order of dict
        iteration determines sheet order in the workbook.
    path:
        Destination ``.xlsx`` path. Parent directories are created.
    index:
        Forwarded to ``pandas.DataFrame.to_excel`` for every sheet.
    **pandas_kwargs:
        Extra keyword arguments forwarded to
        ``pandas.DataFrame.to_excel`` for every sheet.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path) as writer:
        for sheet_name, df in frames.items():
            df.to_pandas().to_excel(
                writer,
                sheet_name=sheet_name,
                index=index,
                **pandas_kwargs,
            )
