"""Centralized figure-display defaults for notebook rendering.

 Inline figures in a Jupyter notebook are constrained by browser width
 (typically ~800-1200 px on a laptop). The constants below pin a
 single, sensible default for **screen display** at high DPI.

 The displayed pixel size is ``figsize × dpi``. At 300 dpi with
 ``figsize=(6, 3.6)`` a single panel renders at 1800×1080 px — sharp
 on modern displays while staying compact enough for inline notebook
 viewing. Use ``plt.savefig(..., dpi=300)`` separately for file export.

 Usage in a notebook cell:

     from app.notebook.plot_style import apply_inline_style
     apply_inline_style()

 This is a no-op outside a notebook (it just sets ``rcParams``).

 Any notebook cell that builds its own figure can use a custom size:

     fig, ax = plt.subplots(figsize=(5, 3))
 """

from __future__ import annotations

import matplotlib.pyplot as plt

# Inline rendering defaults.
# 300 dpi keeps inline figures sharp on modern high-DPI screens.
# Combined with the default figsize below this yields a compact
# 1500×900 px inline image that fits well in a notebook cell.
INLINE_DPI: int = 300

# Standard single-panel size in inches (width, height).
DEFAULT_FIGSIZE: tuple[float, float] = (5.0, 3.0)

# Standard multi-panel (e.g. 2×2 apply_grid) size in inches.
DEFAULT_GRID_FIGSIZE: tuple[float, float] = (7.0, 5.0)


def apply_inline_style() -> None:
    """Pin ``rcParams`` to notebook-friendly values.

    Safe to call multiple times; values are reset each call. Call once at
    the top of any notebook that produces figures, ideally in the same
    cell as ``%matplotlib inline``. Sets ``figure.dpi=300`` and
    ``figure.figsize=(5, 3)`` for a compact, sharp inline display.
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
