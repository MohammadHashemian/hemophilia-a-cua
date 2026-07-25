from pathlib import Path

import polars as pl

from utils.logging import setup_root_logger

__all__ = ["show"]


def show(
    df: pl.DataFrame,
    caption: str | None = None,
    format: dict | None = None,
    store: bool = False,
    options: dict = {},
) -> None:
    """Render a polars DataFrame as a styled HTML table in the notebook.

    The ``.style`` API is a pandas-only feature, so this helper converts
    to pandas internally for rendering. Excel export also goes through
    pandas (``excel_writer``); see ``app.notebook.excel_adapter`` for
    the polars-friendly alternative.

    Args:
        df (pl.DataFrame): DataFrame to display.
        caption (str | None, optional): Table caption. Defaults to None.
        format (dict | None, optional): Per-column format dict. Defaults to None.
        store (bool, optional): If True and ``options["storage"]`` is
            provided, also export to Excel. Defaults to False.
        options (dict, optional): Storage options. See example.
    """
    pdf = df.to_pandas()

    style = pdf.style.set_table_attributes('style="font-size:12px; table-layout:fixed;"')

    if caption:
        style = style.set_caption(caption)

    if format:
        style = style.format(format)

    from IPython.display import display

    display(style)
    logger = setup_root_logger()
    try:
        if store:
            storage = options.get("storage", None)
            override = options.get("override", False)
            if storage:
                path: Path = storage.get("excel_writer", None)
                if not path:
                    logger.warning(
                        "Excel writer path not provided. DataFrame not stored."
                    )
                    return
                if path.exists() and not override:
                    logger.warning(
                        f"File {path} already exists. DataFrame not stored to avoid overwriting."
                    )
                    return
                pdf.to_excel(**storage)
                logger.info(f"DataFrame stored successfully at {path}")
            else:
                logger.warning("Storage options not provided. DataFrame not stored.")
    except Exception as e:
        logger.error(f"Error storing DataFrame: {e}")
