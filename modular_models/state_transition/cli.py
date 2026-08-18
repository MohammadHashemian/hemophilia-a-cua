from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from modular_models.state_transition.analysis import StudyRunner
from modular_models.state_transition.context import StudyContext


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hemophilia state-transition model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing model.json, scenarios.json and references.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Validate all JSON inputs and cross-references")

    base = sub.add_parser("base", help="Run the paired deterministic base case")
    base.add_argument("--patients", type=int, default=10000)
    base.add_argument("--scenario", default="base_case")
    base.add_argument("--output-dir", type=Path, default=Path("outputs/state_transition"))

    convergence = sub.add_parser(
        "convergence", help="Run the pre-specified Monte Carlo convergence check"
    )
    convergence.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000])
    convergence.add_argument("--output-dir", type=Path, default=Path("outputs/state_transition"))

    scenarios = sub.add_parser("scenarios", help="Run every pre-specified structural scenario")
    scenarios.add_argument("--patients", type=int, default=10000)
    scenarios.add_argument("--output-dir", type=Path, default=Path("outputs/state_transition"))

    psa = sub.add_parser("psa", help="Run or resume checkpointed probabilistic analysis")
    psa.add_argument("--iterations", type=int, default=2500)
    psa.add_argument("--patients", type=int, required=True)
    psa.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Worker processes; 0 uses all logical CPU cores",
    )
    psa.add_argument("--batch-size", type=int, default=25)
    psa.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    psa.add_argument("--scenario", default="base_case")
    psa.add_argument("--output-dir", type=Path, default=Path("outputs/state_transition/psa"))

    precision = sub.add_parser(
        "psa-precision",
        help="Compare common PSA draws across candidate inner patient counts",
    )
    precision.add_argument("--iterations", type=int, default=40)
    precision.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 2500, 5000, 10000],
    )
    precision.add_argument("--jobs", type=int, default=0)
    precision.add_argument("--batch-size", type=int, default=24)
    precision.add_argument("--scenario", default="base_case")
    precision.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/state_transition/psa_inner_precision"),
    )

    owsa = sub.add_parser("owsa", help="Run or resume checkpointed one-way analysis")
    owsa.add_argument("--patients", type=int, required=True)
    owsa.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Worker processes; 0 uses all logical CPU cores",
    )
    owsa.add_argument("--scenario", default="base_case")
    owsa.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    owsa.add_argument("--output-dir", type=Path, default=Path("outputs/state_transition/owsa"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = StudyContext.load(args.data_dir)
    if args.command == "validate":
        print(
            f"Validated {len(context.parameters)} parameters, "
            f"{len(context.references)} references and {len(context.scenarios)} scenarios."
        )
        return 0

    runner = StudyRunner(context)
    if args.command == "base":
        from modular_models.state_transition.trace import TraceSession

        trace = TraceSession(max_cycles=3)
        result = runner.compare(
            scenario_id=args.scenario,
            n_patients=args.patients,
            trace=trace,
        )
        output_dir: Path = args.output_dir
        summary = result.analysis_summary()
        _write_json(output_dir / f"{args.scenario}_summary.json", summary)
        trace.write_json(output_dir / f"{args.scenario}_dataflow.json")
        trace.render(output_dir / f"{args.scenario}_dataflow.png")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "convergence":
        records = runner.convergence(args.sizes)
        _write_json(args.output_dir / "convergence.json", [asdict(item) for item in records])
        print(json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2))
        return 0

    if args.command == "scenarios":
        results = runner.run_scenarios(n_patients=args.patients)
        payload = {key: result.analysis_summary() for key, result in results.items()}
        _write_json(args.output_dir / "scenario_analysis.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "psa":
        from modular_models.state_transition.production import PSAConfig, PSAProductionPipeline

        psa_config = PSAConfig(
            iterations=args.iterations,
            n_patients=args.patients,
            output_dir=args.output_dir,
            scenario_id=args.scenario,
            n_jobs=args.jobs,
            batch_size=args.batch_size,
            compute_backend=args.backend,
        )
        frame = PSAProductionPipeline(context, psa_config).run()
        print(f"Completed {frame.height} PSA iterations: {psa_config.output_dir}")
        return 0

    if args.command == "psa-precision":
        from modular_models.state_transition.production import (
            PSAInnerLoopConfig,
            PSAInnerLoopDiagnostic,
        )

        precision_config = PSAInnerLoopConfig(
            population_sizes=tuple(sorted(set(args.sizes))),
            iterations=args.iterations,
            output_dir=args.output_dir,
            scenario_id=args.scenario,
            n_jobs=args.jobs,
            batch_size=args.batch_size,
        )
        frame = PSAInnerLoopDiagnostic(context, precision_config).run()
        print(json.dumps(frame.to_dicts(), ensure_ascii=True, indent=2))
        return 0

    if args.command == "owsa":
        from modular_models.state_transition.production import OWSAConfig, OWSAProductionPipeline

        owsa_config = OWSAConfig(
            n_patients=args.patients,
            output_dir=args.output_dir,
            scenario_id=args.scenario,
            n_jobs=args.jobs,
            compute_backend=args.backend,
        )
        frame = OWSAProductionPipeline(context, owsa_config).run()
        print(f"Completed {frame.height} OWSA endpoint runs: {owsa_config.output_dir}")
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
