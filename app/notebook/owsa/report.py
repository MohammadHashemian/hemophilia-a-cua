"""Display-oriented facade for horizon-specific OWSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from app.notebook.owsa.analysis import (
    base_case,
    parameter_ranges,
    robustness_summary,
    sensitivity_summary,
)
from app.notebook.owsa.plots import all_figures
from app.notebook.owsa.scenarios import OWSARange
from app.notebook.owsa.workflow import load_horizon_results, load_owsa_inputs
from app.notebook.psa.scenarios import HorizonSpec, get_horizon
from app.persistence.context import ModelContext
from utils.path_utils import get_project_root


@dataclass
class OWSAReport:
    horizon: HorizonSpec
    df: pl.DataFrame
    ranges: list[OWSARange]
    wtp: float
    figure_dir: Path

    @classmethod
    def load(cls, horizon: str | HorizonSpec) -> OWSAReport:
        spec = get_horizon(horizon)
        root = get_project_root()
        context = ModelContext.load()
        _scenarios, _parameters, ranges, _context = load_owsa_inputs(
            spec,
            root=root,
            context=context,
        )
        wtp = (
            context.economic_policy.gdp_per_capita.IRR
            * context.economic_policy.wtp_multiplier.rare
        )
        figure_dir = root / "app" / "outputs" / "figures" / "owsa" / spec.directory
        figure_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            horizon=spec,
            df=load_horizon_results(spec),
            ranges=ranges,
            wtp=wtp,
            figure_dir=figure_dir,
        )

    def tables(self) -> dict[str, pl.DataFrame]:
        sensitivity = sensitivity_summary(self.df, self.ranges, wtp=self.wtp)
        return {
            "Parameter ranges": parameter_ranges(self.ranges),
            "Base-case cost effectiveness": base_case(self.df, wtp=self.wtp),
            "OWSA sensitivity ranking": sensitivity,
            "Decision robustness": robustness_summary(sensitivity),
            "ICER tornado exclusions": sensitivity.filter(
                ~pl.col("icer_tornado_valid")
            ).select(
                "parameter",
                "label",
                "low_delta_cost",
                "low_delta_qaly",
                "base_delta_cost",
                "base_delta_qaly",
                "high_delta_cost",
                "high_delta_qaly",
            ),
        }

    def figures(
        self,
        *,
        tables: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, plt.Figure]:
        calculated = self.tables() if tables is None else tables
        return all_figures(
            calculated["OWSA sensitivity ranking"],
            self.horizon,
            wtp=self.wtp,
        )

    def save_figures(
        self,
        figures: dict[str, plt.Figure],
        *,
        dpi: int = 300,
    ) -> dict[str, Path]:
        saved = {}
        for name, figure in figures.items():
            path = self.figure_dir / f"{name}.png"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            saved[name] = path
        return saved
