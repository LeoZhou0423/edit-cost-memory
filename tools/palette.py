"""Shared pastel palette and Matplotlib style for the manuscript figures.

The palette is a single source of truth: every figure script imports from here
so that a colour means the same thing in every panel and in the LaTeX build.
The rose/blue/dark trio already used by `make_fig_v11.py` is kept as the
`ROSE`/`BLUE`/`DARK` aliases so existing figures do not shift.

Ramps
-----
PINK / TEAL / BLUE / LAVENDER : sequential, light to saturated, for heatmaps
                                of one signed-quantity, ordering, and groups
DIVERGING                     : blue - near-white - pink, for signed quantities
                                (the fold-change map) and for "good/bad" grids
CYCLE                         : eight pastels, used in this fixed order so that
                                arm identity is stable across figures

Run a quick visual check of every swatch:

    python tools/palette.py --demo out.png
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Named swatches, sampled from the reference palette sheet
# --------------------------------------------------------------------------- #
PINK = ["#fdf2f7", "#fbc8db", "#f8cbe0", "#ecafc6", "#e797b4", "#df81a5",
        "#c9618a"]
TEAL = ["#eef9fc", "#c9ecf5", "#a3dcec", "#8acfe4", "#6fbfd7", "#56a1b8",
        "#3d8398"]
BLUE = ["#eef5fe", "#dcebfd", "#c4dffc", "#b1d7f2", "#9ecafe", "#97ceff",
        "#71b0fd", "#5293c9", "#3b6f9e"]
LAVENDER = ["#f1ebf7", "#e3d5ed", "#d3c7e0", "#bfc9e4", "#a9b6da"]
CREAM = ["#fffaf7", "#fcece6", "#f7ddd4"]

# The eight-colour categorical cycle, in fixed order.
CYCLE = ["#df81a5", "#56a1b8", "#8ab2d1", "#ecadc4", "#6fbfd7", "#b1d7f2",
         "#e797b4", "#a3dcec"]

# Diverging ramp: low (blue) - centre (near-white warm) - high (pink).
DIVERGING = ["#3b6f9e", "#5293c9", "#71b0fd", "#97ceff", "#b1d7f2", "#dcebfd",
             "#fcece6", "#fce4ee", "#fdc0e1", "#f8cbe0", "#ecafc6", "#df81a5"]

# Paper-level aliases.  These three are the LaTeX build names; do not renumber.
# NB: `BLUE` above is the *ramp list*; the dark reference blue is `BLUE_D`.
ROSE = "#df81a5"     # expensive direction / gate
BLUE_D = "#4e7fa8"   # address / reference
DARK = "#2c6a9b"     # dual / optimal
GREY = "#555555"
INK = "#222222"
FAINT = "#cfcfcf"


# --------------------------------------------------------------------------- #
# Colormaps and style
# --------------------------------------------------------------------------- #
def cmap_seq(name="blue"):
    """Sequential colormap from a named ramp (pink | teal | blue | lavender)."""
    from matplotlib.colors import LinearSegmentedColormap

    ramps = {"pink": PINK, "teal": TEAL, "blue": BLUE, "lavender": LAVENDER,
             "cream": CREAM}
    if name not in ramps:
        raise KeyError(f"unknown ramp {name!r}; have {sorted(ramps)}")
    return LinearSegmentedColormap.from_list(f"pastel_{name}", ramps[name])


def cmap_div():
    """Diverging pink/blue colormap with a near-white centre."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("pastel_div", DIVERGING)


def cmap_flat(colour, n=256):
    """Single-hue colormap from white to `colour`, for one-sided heatmaps.

    `colour` may be a hex string, a ramp name, or a ramp list; a ramp resolves
    to its darkest entry so callers can pass a family and get the strongest end.
    """
    from matplotlib.colors import LinearSegmentedColormap

    ramps = {"pink": PINK, "teal": TEAL, "blue": BLUE, "lavender": LAVENDER,
             "cream": CREAM}
    if isinstance(colour, str) and colour in ramps:
        colour = ramps[colour][-1]
    elif isinstance(colour, (list, tuple)):
        colour = colour[-1]
    return LinearSegmentedColormap.from_list(f"flat_{colour}", ["#ffffff", colour])


def apply_style():
    """Apply the manuscript rcParams.  Matches the LaTeX build: serif, 8 pt."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "axes.edgecolor": GREY,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.titlesize": 8.5,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "figure.dpi": 300,
        # editable text in the vector exports
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    try:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CYCLE)
    except Exception:  # pragma: no cover - very old matplotlib
        pass


def annotate_cell(ax, i, j, text, colour):
    """Cell label for heatmaps: centred, readable on both light and dark cells."""
    ax.text(j, i, text, ha="center", va="center", fontsize=6.4, color=colour)


def _demo(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    ramps = {"pink": PINK, "teal": TEAL, "blue": BLUE, "lavender": LAVENDER}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    for ax, (name, ramp) in zip(axes[:2], ramps.items()):
        ax.imshow(np.arange(len(ramp))[None, :], aspect="auto",
                  cmap=cmap_seq(name), extent=(0, len(ramp), 0, 1))
        ax.set_title(f"{name} ramp")
        ax.set_yticks([])
    axes[2].imshow(np.linspace(-1, 1, 64)[None, :], aspect="auto",
                   cmap=cmap_div(), extent=(-1, 1, 0, 1))
    axes[2].set_title("diverging")
    axes[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("wrote", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="")
    a = ap.parse_args()
    if a.demo:
        _demo(a.demo)
    else:
        print("PINK     ", " ".join(PINK))
        print("TEAL     ", " ".join(TEAL))
        print("BLUE     ", " ".join(BLUE))
        print("LAVENDER ", " ".join(LAVENDER))
        print("DIVERGING", " ".join(DIVERGING))
        print("CYCLE    ", " ".join(CYCLE))
