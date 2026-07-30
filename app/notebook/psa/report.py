"""Facade used by clean, display-oriented PSA analysis notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from app.notebook.psa.analysis import output_dirs, prepare_results
from app.notebook.psa.interpretation import (
    figure_interpretation,
    table_interpretation,
)
from app.notebook.psa.presentation import TableSection, table_sections
from app.notebook.psa.report_plots import all_figures
from app.notebook.psa.scenarios import HorizonSpec, get_horizon
from app.notebook.psa.tables import all_tables
from app.persistence.context import ModelContext


@dataclass
class PSAReport:
    horizon: HorizonSpec
    df: pl.DataFrame
    wtp: float
    figure_dir: Path
    sheet_dir: Path

    @classmethod
    def load(cls, horizon: str | HorizonSpec) -> PSAReport:
        spec = get_horizon(horizon)
        context = ModelContext.load()
        wtp = (
            context.economic_policy.gdp_per_capita.IRR
            * context.economic_policy.wtp_multiplier.rare
        )
        figure_dir, sheet_dir = output_dirs(spec)
        return cls(
            horizon=spec,
            df=prepare_results(spec),
            wtp=wtp,
            figure_dir=figure_dir,
            sheet_dir=sheet_dir,
        )

    @property
    def base_df(self) -> pl.DataFrame:
        return self.df.filter(pl.col("extension").is_null())

    def tables(self) -> dict[str, pl.DataFrame]:
        return all_tables(self.df, wtp=self.wtp)

    def table_sections(
        self,
        *,
        tables: dict[str, pl.DataFrame] | None = None,
    ) -> list[TableSection]:
        """Return narrow presentation views without recalculating results."""

        return table_sections(self.tables() if tables is None else tables)

    def interpret_table(self, section: TableSection) -> str:
        return table_interpretation(section, wtp=self.wtp)

    def interpret_figure(
        self,
        name: str,
        *,
        tables: dict[str, pl.DataFrame] | None = None,
    ) -> str:
        calculated = self.tables() if tables is None else tables
        return figure_interpretation(
            name,
            df=self.df,
            tables=calculated,
            wtp=self.wtp,
        )

    def figures(self) -> dict[str, plt.Figure]:  # type: ignore
        return all_figures(self.df, self.horizon, wtp=self.wtp)

    def save_figures(
        self,
        figures: dict[str, plt.Figure],  # type: ignore
        *,
        dpi: int = 300,
    ) -> dict[str, Path]:
        saved = {}
        for name, figure in figures.items():
            path = self.figure_dir / f"{name}.png"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            saved[name] = path
        return saved
