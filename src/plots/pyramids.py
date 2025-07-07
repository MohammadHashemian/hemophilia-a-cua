from ..data.loaders import load_population_pyramids, PROJECT_ROOT
from ..utils.logger import suppress_matplotlib_debug
from ..processing.distribution_adjuster import adjust_population_pyramid
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# TODO:
# Linear and exponential decay to reduce population life expectancy to 73
def linear_survival_curve(age, maximum_age=100):
    return np.clip((maximum_age - age) / maximum_age, 0, 1)


def plot_pyramids(output_path: Path = PROJECT_ROOT / "outputs" / "figures"):
    # Creating output directory if not exists
    if not output_path.exists():
        output_path.mkdir()
    # Loading population pyramids data
    dfs = load_population_pyramids()

    suppress_matplotlib_debug()
    for year, df in dfs.items():
        max_age = df["Age"].max()
        df = adjust_population_pyramid(df, "M", "Hemophilia", 17.1 / 100000)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["Age"], df["Hemophilia"], alpha=0.7)
        ax.set_xlim(-1, max_age)
        ax.set_ylabel("Population")
        ax.set_xlabel("Age")
        ax.set_title("Iran Severe Hemophilia A Population Pyramid")
        fig.tight_layout()
        fig.savefig(output_path / f"pyramid_{year}.png")


def plot_pyramids_animation(
    output_path: Path = PROJECT_ROOT / "outputs" / "figures",
):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    dfs = load_population_pyramids()
    suppress_matplotlib_debug()

    years = sorted(dfs.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    max_age = max(df["Age"].max() for df in dfs.values())
    max_m = max(df["M"].max() for df in dfs.values())

    ax.set_xlim(-1, max_age + 1)
    ax.set_ylim(0, max_m * 1.1)
    ax.set_xlabel("Age")
    ax.set_ylabel("Population")
    ax.set_title("Iran Severe Hemophilia A Population Pyramid")

    def update(year_idx):
        year = years[year_idx]
        df = dfs[year]
        df = adjust_population_pyramid(df, "M", "Hemophilia", 17.1 / 100000)
        ax.clear()

        ax.bar(df["Age"], df["Hemophilia"], alpha=0.7)
        ax.set_xlim(-1, max_age + 1)
        ax.set_xlabel("Age")
        ax.set_ylabel("Population")
        ax.set_title(f"Iran Population Pyramid - Year {year}")
        blue_patch = mpatches.Patch(color="blue", label=round(df["Hemophilia"].sum()))
        ax.legend(handles=[blue_patch])
        ax.grid(axis="y")

    anim = FuncAnimation(fig, update, frames=len(years), interval=1000, repeat=True)  # type: ignore

    gif_path = output_path / "population_pyramid.gif"
    writer = PillowWriter(fps=3)
    anim.save(gif_path, writer=writer)
    print(f"Animation saved to {gif_path}")


if __name__ == "__main__":
    plot_pyramids()
    plot_pyramids_animation()
