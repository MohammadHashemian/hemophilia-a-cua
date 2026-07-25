from pathlib import Path

import polars as pl

__all__ = ["store"]


def store(df: pl.DataFrame, path: Path, override: bool = False) -> None:
    """Store the given polars DataFrame to a CSV file at the specified path.

    The parent directory is created if it does not exist. The CSV is
    written without a row-index column (polars has no implicit index).
    """
    path = Path(path)  # type: ignore # ensure Path object
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not override:
        print(f"File '{path}' already exists. Set override=True to overwrite.")
        return

    df.write_csv(path)
    print(f"DataFrame successfully stored at '{path}'.")
