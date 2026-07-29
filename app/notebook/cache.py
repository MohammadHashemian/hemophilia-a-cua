"""Resolve the on-disk location of cached simulation results.

The legacy combined PSA notebook and the OWSA simulation notebook write their
combined output to
``app/cache/{psa,owsa}/parquet/all_results_combined.parquet``. The
analysis notebooks (05 / 07) then load from there.

If the combined file is missing — for example because the simulation
was killed mid-run, or the user has not run the simulation yet but
does have a stale ``simulation_output.parquet`` from a previous
run — this helper falls back to that older flat file, so the
analysis notebooks can still render with whatever results are
already on disk.

If neither file is present, raise a clear ``FileNotFoundError`` that
points the user at the simulation notebook they need to run.
"""

from __future__ import annotations

from pathlib import Path

from utils.path_utils import get_project_root

__all__ = ["resolve_cache", "CACHE_RESOLVERS"]


def _fallback_message(name: str) -> str:
    return (
        f"No cached {name} results found. Either:\n"
        f"  - run the appropriate horizon-specific PSA `02_simulation.ipynb`"
        f" or `app/notebooks/04_owsa_simulation.ipynb` to produce\n"
        f"    app/cache/{name}/parquet/all_results_combined.parquet, OR\n"
        f"  - drop a previous run's combined parquet at that path, OR\n"
        f"  - leave app/cache/{ ('simulation' if name == 'psa' else 'owsa') }_output.parquet\n"
        f"    in place (it is auto-detected as a fallback)."
    )


def resolve_cache(
    name: str,
    *,
    root: Path | None = None,
) -> Path:
    """Return the path to the combined results parquet for ``name``
    (``"psa"`` or ``"owsa"``), falling back to a stale top-level
    file if the combined one is missing.
    """
    if name not in ("psa", "owsa"):
        raise ValueError(f"name must be 'psa' or 'owsa', got {name!r}")

    root = root or get_project_root()
    cache_root = root / "app" / "cache" / name
    combined = cache_root / "parquet" / "all_results_combined.parquet"
    if combined.exists():
        return combined

    # Fallback: a previous run's flat output file. PSA and OWSA use
    # different filenames; the mapping keeps the two name spaces
    # independent.
    flat_name = {
        "psa": "simulation_output.parquet",
        "owsa": "owsa_output.parquet",
    }[name]
    flat = root / "app" / "cache" / flat_name
    if flat.exists():
        return flat

    raise FileNotFoundError(_fallback_message(name))


# Convenience constants the notebooks can use.
CACHE_RESOLVERS = {
    "psa": lambda root=None: resolve_cache("psa", root=root),
    "owsa": lambda root=None: resolve_cache("owsa", root=root),
}
