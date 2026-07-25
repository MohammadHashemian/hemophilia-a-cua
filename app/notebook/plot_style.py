"""Centralized figure-display defaults for notebook rendering.

Inline figures in a Jupyter notebook are constrained by browser width
(typically ~800-1200 px on a laptop). Matplotlib's default
``figure.dpi=100`` combined with large ``figsize`` values produces
1500-3000 px inline images that bloat the saved ``.ipynb`` file
(each base64 PNG is hundreds of KB). The constants below pin a
single, sensible default for **screen display**.

Figures that need print-grade quality should be saved to disk via
``plt.savefig(..., dpi=300)`` separately, not rendered inline at
300 dpi.

Usage in a notebook cell:

    from app.notebook.plot_style import apply_inline_style
    apply_inline_style()

This is a no-op outside a notebook (it just sets ``rcParams``).

The helper also exposes a ``figsize`` function that returns a size scaled
to the desired **inline pixel width**, so any notebook cell that builds
its own figure can use:

    fig, ax = plt.subplots(figsize=figsize(width_inches=8.0))
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Inline rendering defaults.
# 150 dpi is a deliberate sweet-spot: 10x6 in -> 1500x900 px, which is
# sharp on standard and Retina/4K displays while keeping the on-disk
# base64 image at a manageable ~120 KB per figure (vs ~30 KB at 80 dpi
# and ~300 KB at 200 dpi). For reference: matplotlib's library default
# is 100 dpi, which is too soft on modern displays; 300 dpi is print
# quality and excessive for screen rendering.
INLINE_DPI: int = 150

# Standard single-panel size in inches (width, height).
DEFAULT_FIGSIZE: tuple[float, float] = (10.0, 6.0)

# Standard multi-panel (e.g. 2x2 apply_grid) size in inches.
DEFAULT_GRID_FIGSIZE: tuple[float, float] = (11.0, 8.0)


def apply_inline_style() -> None:
    """Pin ``rcParams`` to notebook-friendly values.

    Safe to call multiple times; values are reset each call. Call once at
    the top of any notebook that produces figures, ideally in the same
    cell as ``%matplotlib inline``.
    """
    plt.rcParams.update(
        {
            "figure.dpi": INLINE_DPI,
            "savefig.dpi": 200,
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.max_open_warning": 20,
        }
    )


def reset_style() -> None:
    """Restore matplotlib's library defaults (mostly for tests)."""
    plt.rcdefaults()
    plt.rcParams.update({"figure.dpi": INLINE_DPI})


__all__ = [
    "INLINE_DPI",
    "DEFAULT_FIGSIZE",
    "DEFAULT_GRID_FIGSIZE",
    "apply_inline_style",
    "reset_style",
]
