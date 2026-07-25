from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from app.visualization.plots.body_weight import plot_body_weight


def _to_polars(df: pl.DataFrame | Any) -> pl.DataFrame:
    """Accept a polars *or* pandas DataFrame; always return polars.

    Polars-native call paths get a free pass. Pandas inputs are
    converted with ``pl.from_pandas`` so the rest of the helper can
    use polars expressions. The conversion is done once, at the entry
    point of ``apply_grid``, so the plot functions downstream keep
    their existing signatures.
    """
    if isinstance(df, pl.DataFrame):
        return df
    try:
        return pl.from_pandas(df)
    except (ImportError, ValueError, TypeError) as exc:
        raise TypeError(
            "apply_grid expects a polars.DataFrame (or a pandas "
            "DataFrame if pandas is installed). Got "
            f"{type(df).__name__}."
        ) from exc


def apply_grid(
    *,
    row_values: Iterable[Any],
    col_values: Iterable[Any],
    row_name: str,
    col_name: str,
    dataframe,
    figsize=(11, 8),
    gridspec_kw=None,
    title_formatter: Callable[[Any, Any], str] | None = None,
    subplot_kwargs=None,
):
    """Build a grid of subplots, one per ``(row_value, col_value)`` pair.

    The ``dataframe`` is sliced by ``(row_name, col_name)`` and the
    resulting subset is passed to ``func`` as ``sub``. Accepts both
    polars and pandas DataFrames; pandas inputs are converted
    internally so plot functions can keep their polars-friendly
    signatures.
    """
    gridspec_kw = gridspec_kw or {
        "wspace": 0.05,
        "hspace": 0.25,
    }

    subplot_kwargs = subplot_kwargs or {}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            row_values_list = list(row_values)
            col_values_list = list(col_values)

            df = _to_polars(dataframe)

            fig, axes = plt.subplots(
                len(row_values_list),
                len(col_values_list),
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                **subplot_kwargs,
            )

            if len(row_values_list) == 1 and len(col_values_list) == 1:
                axes = [[axes]]
            elif len(row_values_list) == 1:
                axes = [axes]
            elif len(col_values_list) == 1:
                axes = [[ax] for ax in axes]

            for i, row_value in enumerate(row_values_list):
                for j, col_value in enumerate(col_values_list):
                    ax = axes[i][j]

                    sub = df.filter(
                        (pl.col(row_name) == row_value)
                        & (pl.col(col_name) == col_value)
                    )

                    func(
                        ax=ax,
                        sub=sub,
                        row_value=row_value,
                        col_value=col_value,
                        i=i,
                        j=j,
                        *args,
                        **kwargs,
                    )

                    if title_formatter:
                        title = title_formatter(row_value, col_value)
                    else:
                        title = f"{row_value} — {col_value}"

                    ax.set_title(title)  # type: ignore
                    ax.grid(True, alpha=0.3)  # type: ignore
                    ax.set_box_aspect(1)  # type: ignore

            return fig, axes

        return wrapper

    return decorator


class OWSAPlotter:
    @staticmethod
    def plot_owsa_icer_tornado(
        summary: pl.DataFrame,
        filter_horizon: str | None = None,
        style: str = "dual_bars",
    ) -> None:
        """Render an OWSA tornado diagram from a polars summary frame.

        The frame must contain at least the columns ``parameter``,
        ``magnitude``, ``low_icer_change`` and ``high_icer_change``;
        optionally ``time_horizon`` for the ``filter_horizon`` slice.
        """
        data = _to_polars(summary)

        if filter_horizon is not None:
            data = data.filter(pl.col("time_horizon") == filter_horizon)

        data = data.sort("magnitude", descending=True)

        labels = data["parameter"].cast(pl.Utf8).to_list()

        y = np.arange(len(labels))

        low_vals = data["low_icer_change"].to_numpy()
        high_vals = data["high_icer_change"].to_numpy()

        if style == "dual_bars":

            fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))

            ax.barh(
                y - 0.2,
                low_vals,
                height=0.35,
                label="Low",
            )

            ax.barh(
                y + 0.2,
                high_vals,
                height=0.35,
                label="High",
            )

            ax.axvline(0, linewidth=1)

            ax.set_yticks(y)
            ax.set_yticklabels(labels)

            ax.set_xlabel("Δ ICER vs Base Case (IRR/QALY)")

            ax.set_title(
                f"OWSA Tornado Diagram — ICER Sensitivity "
                f"({filter_horizon or 'all horizons'})"
            )

            ax.legend()

        elif style == "errorbar":

            fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))

            mid_points = (low_vals + high_vals) / 2

            lower_errors = np.abs(mid_points - low_vals)
            upper_errors = np.abs(high_vals - mid_points)

            ax.errorbar(
                mid_points,
                y,
                xerr=[lower_errors, upper_errors],
                fmt="o",
                markersize=8,
                capsize=5,
                linewidth=2,
                elinewidth=2,
            )

            ax.axvline(0, linewidth=1)

            ax.set_yticks(y)
            ax.set_yticklabels(labels)

            ax.set_xlabel("Δ ICER vs Base Case (IRR/QALY)")

            ax.set_title(
                f"OWSA Sensitivity — ICER " f"({filter_horizon or 'all horizons'})"
            )

        else:
            raise ValueError(
                f"Unknown style: {style}. " "Use 'dual_bars' or 'errorbar'"
            )

        plt.tight_layout()
        plt.show()


__all__ = [
    "apply_grid",
    "OWSAPlotter",
    "plot_body_weight",
]
