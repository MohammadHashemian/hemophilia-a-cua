from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from modular_models.state_transition.engine import EventRates
from modular_models.state_transition.results import SimulationResult
from modular_models.state_transition.types import Strategy


@dataclass(slots=True)
class RunTrace:
    strategy: str
    n_patients: int
    annual_rates: dict[str, float]
    weekly_poisson_lambdas: dict[str, float]
    cycles: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


class TraceSession:
    """Collect auditable inputs, decisions and outputs for one paired run."""

    def __init__(self, *, max_cycles: int = 3) -> None:
        self.max_cycles = max_cycles
        self.runs: list[RunTrace] = []
        self._active: RunTrace | None = None
        self.parameter_snapshot: dict[str, float] = {}
        self.parameter_provenance: dict[str, dict[str, Any]] = {}
        self.options: dict[str, Any] = {}

    def begin(
        self,
        strategy: Strategy,
        n_patients: int,
        rates: EventRates,
        values: dict[str, float],
        options: dict[str, Any],
        provenance: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.parameter_snapshot = dict(values)
        if provenance is not None:
            self.parameter_provenance = dict(provenance)
        self.options = dict(options)
        run = RunTrace(
            strategy=strategy.value,
            n_patients=n_patients,
            annual_rates={event.value: value for event, value in rates.annual.items()},
            weekly_poisson_lambdas={event.value: value for event, value in rates.weekly.items()},
        )
        self.runs.append(run)
        self._active = run

    def record_cycle(self, **record: Any) -> None:
        if self._active is not None and len(self._active.cycles) < self.max_cycles:
            self._active.cycles.append(record)

    def finish(self, result: SimulationResult) -> None:
        if self._active is not None:
            self._active.output = {
                "summary": dict(result.summary),
                "final_state_counts": dict(result.state_counts),
                "mortality": result.mortality,
            }
        self._active = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.parameter_snapshot,
            "parameter_provenance": self.parameter_provenance,
            "scenario_options": self.options,
            "runs": [asdict(run) for run in self.runs],
            "state_transition_decision_logic": [
                "Start each weekly cycle in the current chronic arthropathy state",
                "Generate recurrent acute event counts from strategy-specific Poisson rates",
                "Update cumulative joint bleeds, Pettersson score and chronic state",
                "For ICH, sample case fatality and persistent sequela conditional on survival",
                "Sample age-specific background death as a competing cause",
                "Assign the earliest within-cycle cause when ICH and background death compete",
                "Accumulate FVIII, IRR cost and QALY only until death",
                "Return survivors to the next weekly cycle; death remains absorbing",
            ],
            "data_flow": [
                "validated reference-based JSON",
                "typed StudyContext",
                "scenario and parameter resolution",
                "annual event-rate partition",
                "weekly Poisson lambdas and age-specific mortality probabilities",
                "within-cycle events, competing mortality and patient-memory update",
                "discounted IRR cost and QALY accumulation",
                "clinical, mortality and incremental economic outcomes",
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> TraceSession:
        """Restore a completed trace so its visual report can be regenerated."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        max_cycles = max(
            (len(run.get("cycles", [])) for run in payload.get("runs", [])),
            default=0,
        )
        session = cls(max_cycles=max_cycles)
        session.parameter_snapshot = dict(payload["inputs"])
        session.parameter_provenance = dict(payload["parameter_provenance"])
        session.options = dict(payload["scenario_options"])
        session.runs = [RunTrace(**run) for run in payload["runs"]]
        return session

    @staticmethod
    def _box(
        ax: Any,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        body: str,
        *,
        edge: str = "#5a87b5",
        title_color: str = "#155b82",
        fontsize: float = 8.6,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor="#ffffff",
            edgecolor=edge,
            linewidth=1.35,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.01,
            y + height - 0.018,
            title,
            fontsize=fontsize + 0.5,
            fontweight="bold",
            color=title_color,
            va="top",
        )
        ax.text(
            x + 0.01,
            y + height - 0.047,
            body,
            fontsize=fontsize,
            color="#263746",
            va="top",
            linespacing=1.35,
        )

    @staticmethod
    def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#6b8298", "lw": 1.35},
        )

    def render(self, path: str | Path) -> Path:
        """Render data flow, transition decisions and mortality audit as PNG."""
        if len(self.runs) < 2:
            raise RuntimeError("A comparison trace needs both treatment strategies")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 14), facecolor="#f7f9fc")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        entry = self.parameter_snapshot["entry_age_years"]
        exit_age = self.parameter_snapshot["exit_age_years"]
        cycles = int((exit_age - entry) * self.parameter_snapshot["cycles_per_year"])
        price = self.parameter_snapshot["factor_price_irr_per_iu"]
        ax.text(
            0.035,
            0.972,
            "State-Transition Microsimulation - Base-Case Audit",
            fontsize=20,
            fontweight="bold",
            color="#17324d",
        )
        ax.text(
            0.035,
            0.945,
            f"Exclusive age horizon: {entry:g} to <{exit_age:g} years | {cycles} weekly cycles "
            f"({exit_age - entry:g} years) | FVIII price: {price:,.0f} IRR/IU",
            fontsize=10.5,
            color="#46627f",
        )

        stages = [
            ("1  INPUT", "JSON values, units,\nreferences and assumptions"),
            ("2  VALIDATE", "Typed immutable context\nand cross-file checks"),
            ("3  RESOLVE", "Base/scenario values,\nannual rates and weekly lambda"),
            ("4  SIMULATE", "Events, competing death,\nmemory and state transitions"),
            ("5  ACCUMULATE", "FVIII, IRR cost, QALY\nand clinical outcomes"),
        ]
        for index, (title, body) in enumerate(stages):
            x = 0.035 + index * 0.193
            self._box(ax, x, 0.85, 0.16, 0.07, title, body)
            if index < len(stages) - 1:
                self._arrow(ax, (x + 0.162, 0.885), (x + 0.19, 0.885))

        ax.text(
            0.035,
            0.815,
            "Weekly state-transition decision structure",
            fontsize=13,
            fontweight="bold",
            color="#17324d",
        )
        decision_nodes = [
            (
                0.035,
                0.68,
                0.15,
                0.105,
                "Alive at cycle start",
                "Current chronic state\n+ cumulative patient memory",
            ),
            (
                0.225,
                0.68,
                0.15,
                0.105,
                "Acute events",
                "Poisson counts + onset\nJoint, non-joint, major, ICH",
            ),
            (
                0.415,
                0.68,
                0.15,
                0.105,
                "Update patient state",
                "Joint bleeds -> PS/state\nICH survivor -> Post-ICH draw",
            ),
            (
                0.605,
                0.68,
                0.16,
                0.105,
                "Competing mortality",
                "ICH fatality + age hazard\nEarliest time sets one cause",
            ),
            (
                0.805,
                0.68,
                0.16,
                0.105,
                "Conditional outcome",
                "Death -> cause + absorb\nPost-ICH -> persist\nAlive -> rewards + next cycle",
            ),
        ]
        for x, y, width, height, title, body in decision_nodes:
            edge = "#b24b45" if title == "Conditional outcome" else "#5a87b5"
            self._box(ax, x, y, width, height, title, body, edge=edge, fontsize=8.1)
        for start, end in [
            ((0.185, 0.732), (0.225, 0.732)),
            ((0.375, 0.732), (0.415, 0.732)),
            ((0.565, 0.732), (0.605, 0.732)),
            ((0.765, 0.732), (0.805, 0.732)),
        ]:
            self._arrow(ax, start, end)

        mortality = self.runs[0].output["mortality"]
        age_specific = mortality["age_specific_background"]
        ax.text(
            0.035,
            0.585,
            "Mortality inputs and derived probabilities",
            fontsize=13,
            fontweight="bold",
            color="#17324d",
        )
        band_labels = {
            "age_1_to_lt5": "Age 1 to <5",
            "age_5_to_lt10": "Age 5 to <10",
            "age_10_to_lt12": "Age 10 to <12",
        }
        for index, (name, values) in enumerate(age_specific.items()):
            body = (
                f"Annual hazard: {values['annual_hazard']:.8f}\n"
                f"Weekly p: {values['weekly_probability']:.10f}\n"
                f"Full-band p*: {values['cumulative_probability_if_alive_for_entire_band']:.6%}\n"
                f"Scheduled cycles: {values['scheduled_cycles']}"
            )
            self._box(
                ax,
                0.035 + index * 0.225,
                0.47,
                0.19,
                0.09,
                band_labels[name],
                body,
                fontsize=8.4,
            )
        ich_fatality = mortality["ich_event_mortality"]["case_fatality_probability_input"]
        sequela_p = self.parameter_snapshot["post_ich_sequela_probability"]
        self._box(
            ax,
            0.72,
            0.47,
            0.245,
            0.09,
            "ICH event probabilities",
            f"Case fatality per ICH: {ich_fatality:.6%}\n"
            f"Post-ICH flag after survived ICH: {sequela_p:.6%}\n"
            "*Band probability assumes survival through the full band.",
            edge="#b7853f",
            title_color="#8a5b17",
            fontsize=8.4,
        )

        ax.text(
            0.035,
            0.435,
            "Base-case outcomes by strategy",
            fontsize=13,
            fontweight="bold",
            color="#17324d",
        )
        colors = {"prophylaxis": "#0b8f87", "on_demand": "#c66a25"}
        for index, run in enumerate(self.runs[:2]):
            x = 0.035 + index * 0.485
            output = run.output["summary"]
            mortality_output = run.output["mortality"]
            overall = mortality_output["overall"]
            ich = mortality_output["ich_event_mortality"]
            rates = "\n".join(
                f"{name.replace('_', ' ')}: {value:.6g}/y  |  "
                f"lambda-w {run.weekly_poisson_lambdas[name]:.6g}"
                for name, value in run.annual_rates.items()
            )
            body = (
                f"{rates}\n\n"
                f"Initial: {overall['initial_patients']:,}  |  Alive: {overall['alive_at_end']:,}  "
                f"|  Deaths: {overall['deaths_total']:,}\n"
                f"Background deaths: {overall['deaths_background']:,} "
                f"({overall['background_mortality_probability']:.4%})  |  "
                f"ICH deaths: {overall['deaths_ich']:,} "
                f"({overall['ich_mortality_probability']:.4%})\n"
                f"All-cause mortality: {overall['all_cause_mortality_probability']:.4%}  |  "
                f"ICH events: {ich['ich_events']:,}  |  observed deaths/ICH: "
                f"{ich['observed_deaths_per_ich_event']:.4%}\n"
                f"Post-ICH flag: {output['post_ich_ever_count']:,} "
                f"({output['post_ich_ever_probability']:.4%})  |  "
                f"surviving-ICH patients: {output['patients_with_surviving_ich_count']:,}\n"
                f"Mean cost: {output['mean_cost_irr']:,.0f} IRR  |  "
                f"Mean QALY: {output['mean_qaly']:.5f}\n"
                f"Mean bleeds: {output['mean_total_bleeds']:.3f}  |  Mean Pettersson: "
                f"{output['mean_pettersson_score']:.3f}"
            )
            self._box(
                ax,
                x,
                0.205,
                0.445,
                0.205,
                run.strategy.replace("_", " ").title(),
                body,
                edge=colors[run.strategy],
                title_color=colors[run.strategy],
                fontsize=8.25,
            )

        ax.text(
            0.035,
            0.16,
            "Mortality by age band and cause (observed counts)",
            fontsize=12,
            fontweight="bold",
            color="#17324d",
        )
        for run_index, run in enumerate(self.runs[:2]):
            mortality_output = run.output["mortality"]["age_specific_background"]
            lines = []
            for name, values in mortality_output.items():
                lines.append(
                    f"{band_labels[name]}: exposure "
                    f"{values['exposed_patient_cycles']:,} patient-cycles | "
                    f"background {values['background_deaths']:,} | ICH {values['ich_deaths']:,} | "
                    f"all-cause {values['all_cause_deaths']:,}"
                )
            ax.text(
                0.035 + run_index * 0.485,
                0.132,
                run.strategy.replace("_", " ").title() + "\n" + "\n".join(lines),
                fontsize=8.5,
                color=colors[run.strategy],
                va="top",
                linespacing=1.35,
            )

        ax.text(
            0.035,
            0.025,
            "Death causes are mutually exclusive and assigned by the earliest within-cycle "
            "death time. "
            "All costs are calculated in Iranian rial (IRR).",
            fontsize=9,
            color="#5b6b7a",
        )
        fig.savefig(target, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return target
