"""Recreate the README chart from one local `check.py --stage all` run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = ["tokens", "block0", "last_tokens", "output", "batch"]
MAX_ABS_ERROR = np.array([4.005e-5, 7.063e-6, 4.768e-5, 5.439e-7, 4.619e-7])
ABS_TOL = np.array([2e-4, 2e-4, 2e-4, 2e-5, 2e-5])


def main():
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.76, bottom=0.22)
    y = np.arange(len(STAGES))
    ax.barh(y, MAX_ABS_ERROR, height=0.53, color="#2068a8",
            label="Maximum absolute error")
    ax.scatter(ABS_TOL, y, marker="|", s=380, linewidth=3,
               color="#ed8b24", label="Absolute tolerance", zorder=3)
    ax.set_yticks(y, STAGES)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 4e-4)
    ax.set_xlabel("Absolute difference from reference (log scale)")
    fig.text(0.16, 0.92, "Image encoder: stage-by-stage numerical check",
             fontsize=15, weight="bold")
    ax.grid(axis="x", which="both", alpha=0.17)
    ax.set_axisbelow(True)
    for yi, error in zip(y, MAX_ABS_ERROR):
        ax.annotate(f"{error:.3e}", (error, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    fig.legend(loc="upper left", bbox_to_anchor=(0.16, 0.87), ncol=2, frameon=False)
    fig.text(0.16, 0.05,
             "preprocess and patches: zero error. Checker also uses relative tolerance (rtol=2e-4).",
             fontsize=9, color="#555555")
    output = Path(__file__).with_name("stage_errors.png")
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
