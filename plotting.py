"""Plot helpers: data-vs-MC stacked plots."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

COLORS = ["tab:red", "tab:orange", "gold", "tab:green", "tab:cyan",
          "tab:blue", "tab:purple", "lightgray", "dimgray"]


def draw(h, ax=None, **kwargs):
    """Draw one hist.Hist as a line histogram."""
    ax = ax or plt.gca()
    ax.stairs(h.values(), h.axes[0].edges, **kwargs)
    ax.set_xlabel(h.axes[0].label)
    ax.set_ylabel("Events")
    if "label" in kwargs:
        ax.legend()
    return ax


def plot_stack(data, mc, logy=True, ax=None):
    """data: hist.Hist or None.  mc: dict name -> hist.Hist, in stacking order (bottom first)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    first = next(iter(mc.values()))
    edges = first.axes[0].edges
    centers = 0.5 * (edges[1:] + edges[:-1])

    base = np.zeros(len(centers))
    colors = COLORS + ["black"] * 100
    for (name, h), color in zip(mc.items(), colors):
        top = base + h.values()
        ax.stairs(top, edges, baseline=base, fill=True, color=color, label=name)
        base = top
    ymax = float(base.max())

    if data is not None:
        d = data.values()
        err = np.sqrt(data.variances())
        m = d > 0
        ax.errorbar(centers[m], d[m], yerr=err[m], fmt="ko", label="Data")
        ymax = max(ymax, float(d.max()))

    if ymax <= 0:
        ymax = 1.0
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(ymax * 1e-4, ymax * 5)
    else:
        ax.set_ylim(0, ymax * 1.4)
    ax.set_xlabel(first.axes[0].label)
    ax.set_ylabel("Events")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.figure.tight_layout()
    return ax


def plot_all(results, pdf=None, logy=True):
    """One Data/MC stack per histogram. `results` = {tag: {"hists": {...}, ...}}. Optional multi-page PDF."""
    out = PdfPages(pdf) if pdf else None
    data = results["Data"]["hists"]
    for name in data:
        mc = {tag: r["hists"][name] for tag, r in results.items() if tag != "Data"}
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_stack(data[name], mc, logy=logy, ax=ax)
        if out:
            out.savefig(fig)
        plt.show()
    if out:
        out.close()
