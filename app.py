"""
Hydro-Studio: Hydrological Visual Analytics & Graphics Engine
Standard: Operational Modular Data Evaluation and Prompt Library — v10
Visual Data Analytics, LLC — Technical Operational Protocol (October 2026)
Zero-Incursion Web Deployment & Publication-Grade Hydroinformatics Engine
"""

import io
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Global Typography (v10 Standard)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hydro-Studio",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "axes.edgecolor": "#000000",
    "axes.linewidth": 1.0,
})

MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
MONTH_CENTERS = [16.0, 46.5, 77.0, 108.0, 138.0, 168.0, 198.5, 229.0, 259.5, 290.0, 321.0, 351.5]

# ---------------------------------------------------------
# USGS 16-Stop Progression Table (v10 Specification)
# ---------------------------------------------------------
USGS_STOPS = [
    (0.000, "#FFFFFF"),  # White (Base Minimum / Lowest Flow)
    (0.125, "#891A10"),  # Dark Red
    (0.179, "#D93427"),  # Crimson
    (0.303, "#EF8834"),  # Coral
    (0.357, "#F2AB3C"),  # Sandy Brown
    (0.428, "#F8D548"),  # Annabanana
    (0.482, "#FDF451"),  # Khaki
    (0.536, "#D1F34E"),  # Green-Yellow
    (0.607, "#81DC42"),  # Yellow-Green
    (0.666, "#5ECB3C"),  # Lime Green
    (0.714, "#4EAC4B"),  # Medium Sea Green
    (0.785, "#39828E"),  # Steel Blue
    (0.839, "#2A63C4"),  # Royal Blue
    (0.893, "#1942B4"),  # Dark Slate Blue
    (0.946, "#0921A3"),  # Deep Royal Blue
    (1.000, "#000094"),  # Midnight Blue (Peak High Flow)
]

usgs_cmap = LinearSegmentedColormap.from_list("USGS_Streamflow", USGS_STOPS, N=256)
usgs_cmap.set_bad(color="#d9d9d9")


# ---------------------------------------------------------
# Global Data Ingestion & Hygiene Parameters (v10 Gate)
# ---------------------------------------------------------
def prepare_pairs_and_dowy(df, date_col, value_col):
    """
    Evaluates dataset against v10 hard gates:
      1. Absolute Physical Bounds: Q >= 0. Excise negative values.
      2. Temporal Continuity: delta_t = 1.0 day. Flag macro-gaps.
      3. Data Provenance: Zero synthetic interpolation.
      4. Standardized 366-Day DOWY: Feb 29 permanently anchored to Day 152.
    """
    df_clean = df.dropna(subset=[date_col, value_col]).copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_clean = df_clean.sort_values(by=date_col).reset_index(drop=True)

    # 1. Physical Bounds
    df_clean = df_clean[df_clean[value_col] >= 0].copy()

    # 2. Temporal Continuity
    dt = df_clean[date_col].diff().dt.total_seconds() / 86400.0
    valid_pair = (dt == 1.0)

    df_clean["Q_t"] = df_clean[value_col].shift(0)
    df_clean["Q_next"] = df_clean[value_col].shift(-1)
    df_clean["valid_pair"] = valid_pair.shift(-1).fillna(False)

    # 3. Standardized Water Year
    month = df_clean[date_col].dt.month
    day = df_clean[date_col].dt.day
    year = df_clean[date_col].dt.year
    df_clean["WaterYear"] = np.where(month >= 10, year + 1, year)

    # Permanent Day 152 anchoring for Feb 29
    ref_leap = pd.date_range("2019-10-01", "2020-09-30", freq="D")
    dowy_lookup = {(d.month, d.day): idx + 1 for idx, d in enumerate(ref_leap)}
    df_clean["Standard_DOWY"] = [dowy_lookup.get((m, d), np.nan) for m, d in zip(month, day)]

    return df_clean


def apply_standard_grid(ax):
    """Applies v10 grid hierarchy: solid black major decade lines, dashed gray minor lines."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#000000", linestyle="-", lw=0.9, zorder=1)
    ax.grid(True, which="minor", color="#808080", linestyle="--", lw=0.5, zorder=1)


def format_base10(val):
    """Formats tick labels in strict base-10 integer/decimal notation."""
    if val < 1:
        return f"{val:g}"
    return f"{int(round(val)):,}"


def draw_leap_adjusted_dividers(ax, years, is_ranked=False):
    """Draws dashed black month dividers accounting for leap water years."""
    pre_mar = [31.5, 61.5, 92.5, 123.5]
    for x in pre_mar:
        ax.axvline(x, color="#000000", linestyle="--", lw=0.75, zorder=3)

    post_mar_nonleap = [151.5, 182.5, 212.5, 243.5, 273.5, 304.5, 335.5]

    for idx, wy in enumerate(years):
        y_bottom = idx - 0.5
        y_top = idx + 0.5
        is_leap = calendar.isleap(int(wy))
        offset = 1.0 if is_leap else 0.0

        for base_x in post_mar_nonleap:
            x = base_x + offset
            ax.plot([x, x], [y_bottom, y_top], color="#000000", linestyle="--", lw=0.75, zorder=3)


# ---------------------------------------------------------
# Modular Visualization Engines (MOD-01 through MOD-12)
# ---------------------------------------------------------

def run_mod01(df, date_col, value_col):
    """MOD-01: Enhanced Flow Duration Curve (eFDC)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    n = len(df_clean)
    sorted_q = np.sort(df_clean[value_col].values)[::-1]
    prob = (np.arange(1, n + 1) / (n + 1)) * 100.0

    df_pairs = df_clean[df_clean["valid_pair"]].copy()
    df_pairs = df_pairs[(df_pairs["Q_t"] > 0) & (df_pairs["Q_next"] > 0)]

    df_pairs["prob_t"] = np.interp(df_pairs["Q_t"], sorted_q[::-1], prob[::-1])
    diff = df_pairs["Q_next"] - df_pairs["Q_t"]

    rising = df_pairs[diff > 0]
    falling = df_pairs[diff < 0]
    steady = df_pairs[diff == 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(rising["prob_t"], rising["Q_next"], color="#1f77b4", s=11, edgecolors="#000000", linewidths=0.4, alpha=0.6, label="Rising Limb (+dQ/dt)", zorder=3)
    ax.scatter(falling["prob_t"], falling["Q_next"], color="#d62728", s=11, edgecolors="#000000", linewidths=0.4, alpha=0.6, label="Falling Limb (-dQ/dt)", zorder=3)
    ax.scatter(steady["prob_t"], steady["Q_next"], color="#808080", s=10, edgecolors="#000000", linewidths=0.4, alpha=0.6, label="Equilibrium (dQ/dt = 0)", zorder=3)
    ax.plot(prob, sorted_q, color="#000000", lw=2.0, label="Standard FDC [Q(t)]", zorder=4)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticks(range(0, 101, 5), minor=True)
    apply_standard_grid(ax)

    ax.set_xlabel("Q(t) exceedance probability (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-01: Enhanced Flow Duration Curve (eFDC)", fontsize=11, fontweight="bold", pad=38)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def run_mod02(df, date_col, value_col):
    """MOD-02: Lag-1 dQ/dt Hydrograph Point Cloud."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()
    df_pairs = df_pairs[(df_pairs["Q_t"] > 0) & (df_pairs["Q_next"] > 0)]

    min_val = min(df_pairs["Q_t"].min(), df_pairs["Q_next"].min())
    max_val = max(df_pairs["Q_t"].max(), df_pairs["Q_next"].max())
    log_min = 10 ** np.floor(np.log10(min_val))
    log_max = 10 ** np.ceil(np.log10(max_val))

    fig, ax = plt.subplots(figsize=(6, 6))

    diff = df_pairs["Q_next"] - df_pairs["Q_t"]
    rising = df_pairs[diff > 0]
    falling = df_pairs[diff < 0]
    eq = df_pairs[diff == 0]

    ax.scatter(rising["Q_t"], rising["Q_next"], color="#1f77b4", edgecolors="#000000", linewidths=0.35, s=12, alpha=0.5, label="Rising Limb (+dQ/dt)", zorder=3)
    ax.scatter(falling["Q_t"], falling["Q_next"], color="#d62728", edgecolors="#000000", linewidths=0.35, s=12, alpha=0.5, label="Falling Limb (-dQ/dt)", zorder=3)
    ax.scatter(eq["Q_t"], eq["Q_next"], color="#808080", edgecolors="#000000", linewidths=0.35, s=12, alpha=0.5, label="Steady State (dQ/dt = 0)", zorder=3)

    # 1:1 Identity Line
    ax.plot([log_min, log_max], [log_min, log_max], color="#000000", lw=1.5, linestyle="-", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(log_min, log_max)
    ax.set_ylim(log_min, log_max)
    ax.set_aspect("equal", adjustable="box")

    for a in [ax.xaxis, ax.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        a.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    apply_standard_grid(ax)
    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-02: Lag-1 dQ/dt Hydrograph Point Cloud", fontsize=11, fontweight="bold", pad=38)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def _get_transition_counts(df_clean, value_col, delta_log10=0.25):
    """Helper to partition streamflow into log-bins and return 2D transition counts."""
    df_pairs = df_clean[df_clean["valid_pair"]].copy()
    df_pairs = df_pairs[(df_pairs["Q_t"] > 0) & (df_pairs["Q_next"] > 0)]

    min_val = min(df_pairs["Q_t"].min(), df_pairs["Q_next"].min())
    max_val = max(df_pairs["Q_t"].max(), df_pairs["Q_next"].max())
    dec_min = int(np.floor(np.log10(min_val)))
    dec_max = int(np.ceil(np.log10(max_val)))

    bins_log = np.arange(dec_min, dec_max + delta_log10, delta_log10)
    bins = 10 ** bins_log
    counts, _, _ = np.histogram2d(df_pairs["Q_t"], df_pairs["Q_next"], bins=bins)
    return counts, bins, bins_log, dec_min, dec_max


def run_mod03(df, date_col, value_col):
    """MOD-03: Discrete Transition Matrix (Raw Counts)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    counts, bins, bins_log, dec_min, dec_max = _get_transition_counts(df_clean, value_col, delta_log10=0.25)
    n_bins = len(bins) - 1

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** dec_min, 10 ** dec_max)
    ax.set_ylim(10 ** dec_min, 10 ** dec_max)

    for a in [ax.xaxis, ax.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        a.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    apply_standard_grid(ax)

    for b in bins:
        ax.axvline(b, color="#e0e0e0", lw=0.4, zorder=2)
        ax.axhline(b, color="#e0e0e0", lw=0.4, zorder=2)

    ax.plot([10 ** dec_min, 10 ** dec_max], [10 ** dec_min, 10 ** dec_max], color="#000000", lw=1.2, linestyle="-", zorder=3)

    for i in range(n_bins):
        for j in range(n_bins):
            cnt = int(counts[i, j])
            if cnt > 0:
                xc = 10 ** ((bins_log[i] + bins_log[i + 1]) / 2.0)
                yc = 10 ** ((bins_log[j] + bins_log[j + 1]) / 2.0)
                if j > i:
                    ax.text(xc, yc, f"{cnt}", ha="center", va="center", color="#1f77b4", fontweight="bold", fontsize=6, zorder=5)
                elif j < i:
                    ax.text(xc, yc, f"{cnt}", ha="center", va="center", color="#d62728", fontweight="bold", fontsize=6, zorder=5)
                else:
                    ax.text(xc, yc, f"{cnt}", ha="center", va="center", color="#000000", fontweight="bold", fontsize=6, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))

    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-03: Discrete Transition Matrix (Raw Counts)", fontsize=11, fontweight="bold", pad=38)

    fig.tight_layout()
    return fig


def run_mod04(df, date_col, value_col):
    """MOD-04: Conditional Forecast Probability Matrix (Column-Normalized / Vertical)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    counts, bins, bins_log, dec_min, dec_max = _get_transition_counts(df_clean, value_col, delta_log10=0.25)
    n_bins = len(bins) - 1

    col_sums = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.where(col_sums[:, None] > 0, (counts / col_sums[:, None]) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** dec_min, 10 ** dec_max)
    ax.set_ylim(10 ** dec_min, 10 ** dec_max)

    for a in [ax.xaxis, ax.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        a.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    apply_standard_grid(ax)

    for b in bins:
        ax.axvline(b, color="#e0e0e0", lw=0.4, zorder=2)
        ax.axhline(b, color="#e0e0e0", lw=0.4, zorder=2)

    ax.plot([10 ** dec_min, 10 ** dec_max], [10 ** dec_min, 10 ** dec_max], color="#000000", lw=1.2, zorder=3)

    for i in range(n_bins):
        for j in range(n_bins):
            val = prob_matrix[i, j]
            if val > 0:
                xc = 10 ** ((bins_log[i] + bins_log[i + 1]) / 2.0)
                yc = 10 ** ((bins_log[j] + bins_log[j + 1]) / 2.0)
                txt = f"{val:.1f}%" if val >= 1.0 else f"{val:.2f}%"
                if j > i:
                    ax.text(xc, yc, txt, ha="center", va="center", color="#1f77b4", fontweight="bold", fontsize=5.5, zorder=5)
                elif j < i:
                    ax.text(xc, yc, txt, ha="center", va="center", color="#d62728", fontweight="bold", fontsize=5.5, zorder=5)
                else:
                    ax.text(xc, yc, txt, ha="center", va="center", color="#000000", fontweight="bold", fontsize=5.5, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))

    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-04: Conditional Forecast Probability Matrix [P(Q_t+1 | Q_t)]", fontsize=11, fontweight="bold", pad=38)

    fig.tight_layout()
    return fig


def run_mod05(df, date_col, value_col):
    """MOD-05: Conditional Antecedent Probability Matrix (Row-Normalized / Horizontal)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    counts, bins, bins_log, dec_min, dec_max = _get_transition_counts(df_clean, value_col, delta_log10=0.25)
    n_bins = len(bins) - 1

    row_sums = counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.where(row_sums[None, :] > 0, (counts / row_sums[None, :]) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** dec_min, 10 ** dec_max)
    ax.set_ylim(10 ** dec_min, 10 ** dec_max)

    for a in [ax.xaxis, ax.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        a.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    apply_standard_grid(ax)

    for b in bins:
        ax.axvline(b, color="#e0e0e0", lw=0.4, zorder=2)
        ax.axhline(b, color="#e0e0e0", lw=0.4, zorder=2)

    ax.plot([10 ** dec_min, 10 ** dec_max], [10 ** dec_min, 10 ** dec_max], color="#000000", lw=1.2, zorder=3)

    for i in range(n_bins):
        for j in range(n_bins):
            val = prob_matrix[i, j]
            if val > 0:
                xc = 10 ** ((bins_log[i] + bins_log[i + 1]) / 2.0)
                yc = 10 ** ((bins_log[j] + bins_log[j + 1]) / 2.0)
                txt = f"{val:.1f}%" if val >= 1.0 else f"{val:.2f}%"
                if j > i:
                    ax.text(xc, yc, txt, ha="center", va="center", color="#1f77b4", fontweight="bold", fontsize=5.5, zorder=5)
                elif j < i:
                    ax.text(xc, yc, txt, ha="center", va="center", color="#d62728", fontweight="bold", fontsize=5.5, zorder=5)
                else:
                    ax.text(xc, yc, txt, ha="center", va="center", color="#000000", fontweight="bold", fontsize=5.5, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))

    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-05: Conditional Antecedent Probability Matrix [P(Q_t | Q_t+1)]", fontsize=11, fontweight="bold", pad=38)

    fig.tight_layout()
    return fig


def run_mod06(df, date_col, value_col):
    """MOD-06: Chronological Raster Hydrograph."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(columns=range(1, 367)).sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    valid_flows = df_clean[df_clean[value_col] > 0][value_col]
    vmin = 10 ** np.floor(np.log10(np.maximum(0.01, valid_flows.min())))
    vmax = 10 ** np.ceil(np.log10(valid_flows.max()))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    years = pivot.index.values

    mesh = ax.imshow(
        np.maximum(0.001, pivot.values),
        aspect="auto",
        cmap=usgs_cmap,
        norm=norm,
        origin="lower",
        extent=[0.5, 366.5, -0.5, len(years) - 0.5],
        interpolation="nearest",
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Flow (cfs)", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    draw_leap_adjusted_dividers(ax, years, is_ranked=False)

    y_indices = np.arange(len(years))
    major_mask = [(y % 5 == 0) for y in years]
    ax.set_yticks(y_indices[major_mask])
    ax.set_yticklabels(years[major_mask], fontsize=9)
    ax.set_yticks(y_indices, minor=True)
    ax.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8, zorder=2)
    ax.grid(False, which="minor", axis="y")

    ax.set_xlim(0.5, 366.5)
    ax.set_xticks(range(30, 366, 30))
    ax.set_xticks(range(10, 366, 10), minor=True)
    ax.set_xlabel("Day of Water Year", fontsize=10, fontweight="bold")
    ax.set_ylabel("Water Year", fontsize=10, fontweight="bold")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(MONTH_CENTERS)
    ax_top.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax_top.tick_params(length=0, pad=4)

    ax.set_title("MOD-06: Chronological Raster Hydrograph", fontsize=12, fontweight="bold", pad=38)
    fig.tight_layout()
    return fig


def run_mod07(df, date_col, value_col):
    """MOD-07: Volumetric Raster Hydrograph (Rank-Ordered)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    wy_totals = df_clean.groupby("WaterYear")[value_col].sum()
    ranked_wy_descending = wy_totals.sort_values(ascending=False).index.values
    n_years = len(ranked_wy_descending)

    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(index=ranked_wy_descending, columns=range(1, 367))

    fig, ax = plt.subplots(figsize=(12, 8))

    valid_flows = df_clean[df_clean[value_col] > 0][value_col]
    vmin = 10 ** np.floor(np.log10(np.maximum(0.01, valid_flows.min())))
    vmax = 10 ** np.ceil(np.log10(valid_flows.max()))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    mesh = ax.imshow(
        np.maximum(0.001, pivot.values),
        aspect="auto",
        cmap=usgs_cmap,
        norm=norm,
        origin="upper",
        extent=[0.5, 366.5, n_years + 0.5, 0.5],
        interpolation="nearest",
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Flow (cfs)", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    for idx, wy in enumerate(ranked_wy_descending):
        rank = idx + 1
        y_top = rank - 0.5
        y_bottom = rank + 0.5
        is_leap = calendar.isleap(int(wy))
        offset = 1.0 if is_leap else 0.0

        for x in [31.5, 61.5, 92.5, 123.5]:
            ax.plot([x, x], [y_top, y_bottom], color="#000000", linestyle="--", lw=0.75, zorder=3)

        for base_x in [151.5, 182.5, 212.5, 243.5, 273.5, 304.5, 335.5]:
            x = base_x + offset
            ax.plot([x, x], [y_top, y_bottom], color="#000000", linestyle="--", lw=0.75, zorder=3)

    major_ranks = [1] + [r for r in range(10, n_years + 1, 10)]
    ax.set_yticks(major_ranks)
    ax.set_yticklabels(major_ranks, fontsize=9)
    ax.set_yticks(range(1, n_years + 1), minor=True)
    ax.set_ylim(n_years + 0.5, 0.5)

    ax.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8, zorder=2)
    ax.grid(False, which="minor", axis="y")

    ax.set_xlim(0.5, 366.5)
    ax.set_xticks(range(50, 366, 50))
    ax.set_xticks(range(10, 366, 10), minor=True)
    ax.set_xlabel("Day of Water Year", fontsize=10, fontweight="bold")
    ax.set_ylabel("Volumetric order (1 = greatest)", fontsize=10, fontweight="bold")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(MONTH_CENTERS)
    ax_top.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax_top.tick_params(length=0, pad=4)

    ax.set_title("MOD-07: Volumetric Raster Hydrograph (Rank-Ordered)", fontsize=12, fontweight="bold", pad=38)
    fig.tight_layout()
    return fig


def run_mod08(df, date_col, value_col):
    """MOD-08: Annual FDC Spaghetti Plot based on Water Year."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    
    n_total = len(df_clean)
    sorted_total = np.sort(df_clean[value_col].values)[::-1]
    prob_total = (np.arange(1, n_total + 1) / (n_total + 1)) * 100.0

    years = np.sort(df_clean["WaterYear"].unique())
    min_wy, max_wy = years.min(), years.max()

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("Spectral")
    norm = Normalize(vmin=min_wy, vmax=max_wy)

    for wy in years:
        sub = df_clean[df_clean["WaterYear"] == wy]
        if len(sub) < 30:
            continue
        n_wy = len(sub)
        sorted_wy = np.sort(sub[value_col].values)[::-1]
        prob_wy = (np.arange(1, n_wy + 1) / (n_wy + 1)) * 100.0
        ax.plot(prob_wy, sorted_wy, color=cmap(norm(wy)), lw=0.9, alpha=0.35, zorder=2)

    ax.plot(prob_total, sorted_total, color="#000000", lw=2.5, label="Period of Record", zorder=4)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticks(range(0, 101, 5), minor=True)
    apply_standard_grid(ax)

    ax.set_xlabel("Exceedance Probability (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-08: Annual FDC Spaghetti Plot based on Water Year", fontsize=11, fontweight="bold", pad=38)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=8.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label("Water Year", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    fig.tight_layout()
    return fig


def _calc_annual_percentiles(df_clean, value_col):
    """Computes empirical Q10, Q50, and Q90 values per independent Water Year."""
    records = []
    for wy, group in df_clean.groupby("WaterYear"):
        if len(group) < 30:
            continue
        vals = group[value_col].values
        q10 = np.percentile(vals, 90)  # Exceeded 10%
        q50 = np.percentile(vals, 50)  # Exceeded 50%
        q90 = np.percentile(vals, 10)  # Exceeded 90%
        vol = vals.sum()
        records.append({"WaterYear": int(wy), "Q10": q10, "Q50": q50, "Q90": q90, "AnnualVol": vol})
    return pd.DataFrame(records)


def run_mod09(df, date_col, value_col):
    """MOD-09: Annual FDC Threshold Trends (Chronological) with 10x Headroom."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_trends = _calc_annual_percentiles(df_clean, value_col).sort_values("WaterYear")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df_trends["WaterYear"], df_trends["Q10"], color="#0000FF", s=50, edgecolors="#000000", linewidths=0.75, label="Q10 (High Flow)", zorder=3)
    ax.scatter(df_trends["WaterYear"], df_trends["Q50"], color="#2ca02c", s=50, edgecolors="#000000", linewidths=0.75, label="Q50 (Median Flow)", zorder=3)
    ax.scatter(df_trends["WaterYear"], df_trends["Q90"], color="#FF0000", s=50, edgecolors="#000000", linewidths=0.75, label="Q90 (Low Flow)", zorder=3)

    ax.set_yscale("log")
    min_q = df_trends[["Q10", "Q50", "Q90"]].min().min()
    max_q = df_trends[["Q10", "Q50", "Q90"]].max().max()

    # 10x upper headroom rule per v10 protocol
    y_min_bound = 10 ** np.floor(np.log10(max(0.01, min_q * 0.5)))
    y_max_bound = 10 ** (np.ceil(np.log10(max_q)) + 1.0)
    ax.set_ylim(y_min_bound, y_max_bound)

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    min_wy = df_trends["WaterYear"].min()
    max_wy = df_trends["WaterYear"].max()
    ax.set_xlim(min_wy - 1, max_wy + 1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    apply_standard_grid(ax)

    ax.set_xlabel("Water Year", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-09: Annual FDC Threshold Trends (Chronological)", fontsize=11, fontweight="bold", pad=38)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=8, edgecolor="#808080")

    fig.tight_layout()
    return fig


def run_mod10(df, date_col, value_col):
    """MOD-10: Annual FDC Volumetric Thresholds (Rank-Ordered) with 10x Headroom."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_trends = _calc_annual_percentiles(df_clean, value_col)
    df_ranked = df_trends.sort_values("AnnualVol", ascending=False).reset_index(drop=True)
    df_ranked["VolumetricRank"] = df_ranked.index + 1
    n_ranks = len(df_ranked)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df_ranked["VolumetricRank"], df_ranked["Q10"], color="#0000FF", s=50, edgecolors="#000000", linewidths=0.75, label="Q10 (High Flow)", zorder=3)
    ax.scatter(df_ranked["VolumetricRank"], df_ranked["Q50"], color="#2ca02c", s=50, edgecolors="#000000", linewidths=0.75, label="Q50 (Median Flow)", zorder=3)
    ax.scatter(df_ranked["VolumetricRank"], df_ranked["Q90"], color="#FF0000", s=50, edgecolors="#000000", linewidths=0.75, label="Q90 (Low Flow)", zorder=3)

    ax.set_yscale("log")
    min_q = df_ranked[["Q10", "Q50", "Q90"]].min().min()
    max_q = df_ranked[["Q10", "Q50", "Q90"]].max().max()

    y_min_bound = 10 ** np.floor(np.log10(max(0.01, min_q * 0.5)))
    y_max_bound = 10 ** (np.ceil(np.log10(max_q)) + 1.0)
    ax.set_ylim(y_min_bound, y_max_bound)

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    ax.set_xlim(0.5, n_ranks + 0.5)
    major_ticks = [1] + [r for r in range(10, n_ranks + 1, 10)]
    ax.set_xticks(major_ticks)
    ax.set_xticks(range(1, n_ranks + 1), minor=True)

    apply_standard_grid(ax)

    ax.set_xlabel("Annual Volume Rank (1 = Largest Volume)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-10: Annual FDC Volumetric Thresholds (Rank-Ordered)", fontsize=11, fontweight="bold", pad=38)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=8, edgecolor="#808080")

    fig.tight_layout()
    return fig


def run_mod11(df, date_col, value_col):
    """MOD-11: Chronological Volumetric Rank Scatterplot."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_trends = _calc_annual_percentiles(df_clean, value_col)
    df_ranked = df_trends.sort_values("AnnualVol", ascending=False).reset_index(drop=True)
    df_ranked["VolumetricRank"] = df_ranked.index + 1
    n_years = len(df_ranked)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df_ranked["WaterYear"], df_ranked["VolumetricRank"], color="#000080", s=40, edgecolors="#000000", linewidths=0.75, zorder=3)

    ax.set_ylim(n_years + 0.5, 0.5)  # Invert so Rank 1 is at top
    major_y = [1] + [r for r in range(10, n_years + 1, 10)]
    ax.set_yticks(major_y)
    ax.set_yticks(range(1, n_years + 1), minor=True)

    min_wy = df_ranked["WaterYear"].min()
    max_wy = df_ranked["WaterYear"].max()
    ax.set_xlim(min_wy - 1, max_wy + 1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    apply_standard_grid(ax)

    ax.set_xlabel("Water Year", fontsize=10, fontweight="bold")
    ax.set_ylabel("Volumetric Rank (1 = Largest Volume)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-11: Chronological Volumetric Rank Scatterplot", fontsize=11, fontweight="bold", pad=38)

    fig.tight_layout()
    return fig


def run_mod12(df, date_col, value_col):
    """
    MOD-12: Composite Hydroinformatics Dashboard.
    Synchronizes:
      - Panel A (Top-Left): MOD-01 Enhanced Flow Duration Curve
      - Panel B (Top-Right): MOD-02 Lag-1 Differential Phase Space (Aspect 1:1)
      - Panel C (Bottom-Left): MOD-06 Chronological Raster Hydrograph
      - Panel D (Bottom-Right): MOD-09 Annual FDC Threshold Trends (Chronological)
    """
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()
    df_pairs = df_pairs[(df_pairs["Q_t"] > 0) & (df_pairs["Q_next"] > 0)]

    min_v = min(df_pairs["Q_t"].min(), df_pairs["Q_next"].min())
    max_v = max(df_pairs["Q_t"].max(), df_pairs["Q_next"].max())
    dec_min = int(np.floor(np.log10(min_v)))
    dec_max = int(np.ceil(np.log10(max_v)))
    log_min = 10 ** dec_min
    log_max = 10 ** dec_max

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # -----------------------------------------------------
    # Panel A: MOD-01 eFDC
    # -----------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    n_a = len(df_clean)
    sorted_q_a = np.sort(df_clean[value_col].values)[::-1]
    prob_a = (np.arange(1, n_a + 1) / (n_a + 1)) * 100.0

    df_pairs_a = df_pairs.copy()
    df_pairs_a["prob_t"] = np.interp(df_pairs_a["Q_t"], sorted_q_a[::-1], prob_a[::-1])
    diff_a = df_pairs_a["Q_next"] - df_pairs_a["Q_t"]

    ax_a.scatter(df_pairs_a[diff_a > 0]["prob_t"], df_pairs_a[diff_a > 0]["Q_next"], color="#1f77b4", s=8, alpha=0.5, label="+dQ/dt", zorder=3)
    ax_a.scatter(df_pairs_a[diff_a < 0]["prob_t"], df_pairs_a[diff_a < 0]["Q_next"], color="#d62728", s=8, alpha=0.5, label="-dQ/dt", zorder=3)
    ax_a.plot(prob_a, sorted_q_a, color="#000000", lw=1.5, zorder=4)

    ax_a.set_yscale("log")
    ax_a.set_xlim(0, 100)
    ax_a.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))
    apply_standard_grid(ax_a)
    ax_a.set_title("A  MOD-01: Enhanced Flow Duration Curve", loc="left", fontsize=10, fontweight="bold", pad=20)
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, fontsize=7.5)
    ax_a.set_xlabel("Exceedance Probability (%)", fontsize=8, fontweight="bold")
    ax_a.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=8, fontweight="bold")

    # -----------------------------------------------------
    # Panel B: MOD-02 Lag-1 Phase Space (Aspect = 1:1)
    # -----------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(df_pairs[diff_a > 0]["Q_t"], df_pairs[diff_a > 0]["Q_next"], color="#1f77b4", s=8, alpha=0.5, label="+dQ/dt", zorder=3)
    ax_b.scatter(df_pairs[diff_a < 0]["Q_t"], df_pairs[diff_a < 0]["Q_next"], color="#d62728", s=8, alpha=0.5, label="-dQ/dt", zorder=3)
    ax_b.scatter(df_pairs[diff_a == 0]["Q_t"], df_pairs[diff_a == 0]["Q_next"], color="#808080", s=8, alpha=0.5, label="dQ/dt=0", zorder=3)
    ax_b.plot([log_min, log_max], [log_min, log_max], color="#000000", lw=1.2, zorder=4)

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(log_min, log_max)
    ax_b.set_ylim(log_min, log_max)
    ax_b.set_aspect("equal", adjustable="box")

    for a in [ax_b.xaxis, ax_b.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    apply_standard_grid(ax_b)
    ax_b.set_title("B  MOD-02: Lag-1 Differential Phase Space", loc="left", fontsize=10, fontweight="bold", pad=20)
    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False, fontsize=7.5)
    ax_b.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=8, fontweight="bold")
    ax_b.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=8, fontweight="bold")

    # -----------------------------------------------------
    # Panel C: MOD-06 Chronological Raster Hydrograph
    # -----------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(columns=range(1, 367)).sort_index(ascending=True)
    years = pivot.index.values

    norm_c = LogNorm(vmin=log_min, vmax=log_max)
    mesh_c = ax_c.imshow(
        np.maximum(0.001, pivot.values),
        aspect="auto",
        cmap=usgs_cmap,
        norm=norm_c,
        origin="lower",
        extent=[0.5, 366.5, -0.5, len(years) - 0.5],
        interpolation="nearest",
    )

    draw_leap_adjusted_dividers(ax_c, years, is_ranked=False)

    y_indices = np.arange(len(years))
    major_mask = [(y % 5 == 0) for y in years]
    ax_c.set_yticks(y_indices[major_mask])
    ax_c.set_yticklabels(years[major_mask], fontsize=7.5)
    ax_c.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8, zorder=2)

    ax_c.set_xlim(0.5, 366.5)
    ax_c.set_xticks(range(50, 366, 50))
    ax_c.set_xlabel("Day of Water Year", fontsize=8, fontweight="bold")
    ax_c.set_ylabel("Water Year", fontsize=8, fontweight="bold")

    cbar_c = fig.colorbar(mesh_c, ax=ax_c, pad=0.02)
    cbar_c.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    cbar_c.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))
    cbar_c.set_label("Flow (cfs)", fontsize=7.5, fontweight="bold")

    ax_c.set_title("C  MOD-06: Chronological Raster Hydrograph", loc="left", fontsize=10, fontweight="bold", pad=15)

    # -----------------------------------------------------
    # Panel D: MOD-09 Chronological Threshold Trends
    # -----------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    df_trends = _calc_annual_percentiles(df_clean, value_col).sort_values("WaterYear")

    ax_d.scatter(df_trends["WaterYear"], df_trends["Q10"], color="#0000FF", s=20, edgecolors="#000000", linewidths=0.5, label="Q10", zorder=3)
    ax_d.scatter(df_trends["WaterYear"], df_trends["Q50"], color="#2ca02c", s=20, edgecolors="#000000", linewidths=0.5, label="Q50", zorder=3)
    ax_d.scatter(df_trends["WaterYear"], df_trends["Q90"], color="#FF0000", s=20, edgecolors="#000000", linewidths=0.5, label="Q90", zorder=3)

    ax_d.set_yscale("log")
    ax_d.set_ylim(log_min, log_max * 10)  # Headroom
    ax_d.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_base10(y)))

    ax_d.set_xlim(df_trends["WaterYear"].min() - 1, df_trends["WaterYear"].max() + 1)
    ax_d.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    apply_standard_grid(ax_d)
    ax_d.set_title("D  MOD-09: Annual Threshold Trends (Q10, Q50, Q90)", loc="left", fontsize=10, fontweight="bold", pad=15)
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=7.5)
    ax_d.set_xlabel("Water Year", fontsize=8, fontweight="bold")
    ax_d.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=8, fontweight="bold")

    fig.suptitle("MOD-12: Composite Hydroinformatics Dashboard", fontsize=13, fontweight="bold", y=0.99)
    return fig


MODULE_REGISTRY = {
    "MOD-01: Enhanced Flow Duration Curve (eFDC)": run_mod01,
    "MOD-02: Lag-1 dQ/dt Hydrograph Point Cloud": run_mod02,
    "MOD-03: Discrete Transition Matrix (Raw Counts)": run_mod03,
    "MOD-04: Conditional Forecast Probability Matrix": run_mod04,
    "MOD-05: Conditional Antecedent Probability Matrix": run_mod05,
    "MOD-06: Chronological Raster Hydrograph": run_mod06,
    "MOD-07: Volumetric Raster Hydrograph (Rank-Ordered)": run_mod07,
    "MOD-08: Annual FDC Spaghetti Plot based on Water Year": run_mod08,
    "MOD-09: Annual FDC Threshold Trends (Chronological)": run_mod09,
    "MOD-10: Annual FDC Volumetric Thresholds (Rank-Ordered)": run_mod10,
    "MOD-11: Chronological Volumetric Rank Scatterplot": run_mod11,
    "MOD-12: Composite Hydroinformatics Dashboard": run_mod12,
}


# ---------------------------------------------------------
# Streamlit Web User Interface
# ---------------------------------------------------------
def main():
    st.sidebar.title("Hydro-Studio")
    st.sidebar.markdown("**Operational Modular Visual Analytics (v10)**")
    st.sidebar.caption("© Visual Data Analytics, LLC (October 2026)")
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Streamflow CSV",
        type=["csv"],
        help="Upload standard CSV with chronological streamflow data."
    )

    if uploaded_file is None:
        st.info("👋 Welcome to Hydro-Studio. Please upload a streamflow CSV file in the sidebar to begin analysis.")
        return

    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
        return

    cols = list(df_raw.columns)
    if len(cols) < 2:
        st.error("The CSV file must contain at least two columns: Date and Streamflow.")
        return

    st.sidebar.subheader("Dataset Mapping")
    default_date_idx = 0
    default_val_idx = 1 if len(cols) > 1 else 0

    for idx, c in enumerate(cols):
        if "date" in str(c).lower() or "time" in str(c).lower():
            default_date_idx = idx
            break

    for idx, c in enumerate(cols):
        if any(term in str(c).lower() for term in ["flow", "discharge", "q", "00060"]):
            default_val_idx = idx
            break

    date_col = st.sidebar.selectbox("Date / Timestamp Column", options=cols, index=default_date_idx)
    value_col = st.sidebar.selectbox("Discharge Column (cfs)", options=cols, index=default_val_idx)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Module Selection")
    module_choice = st.sidebar.selectbox("Select Visual Module", options=list(MODULE_REGISTRY.keys()), index=5)

    st.sidebar.markdown("---")
    generate_btn = st.sidebar.button("⚡ Generate Plot", type="primary", use_container_width=True)

    st.subheader(module_choice)

    if generate_btn or "current_fig" in st.session_state:
        if generate_btn:
            with st.spinner("Processing time-series matrices and executing rendering..."):
                func = MODULE_REGISTRY[module_choice]
                fig = func(df_raw, date_col, value_col)
                st.session_state["current_fig"] = fig
                st.session_state["current_module"] = module_choice

        fig = st.session_state.get("current_fig")
        if fig is not None:
            st.pyplot(fig, clear_figure=False)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)

            mod_id = st.session_state.get("current_module", "Plot").split(":")[0].strip()
            st.download_button(
                label="📥 Download High-Resolution Plot (PNG - 300 DPI)",
                data=buf,
                file_name=f"{mod_id}_Publication_Grade.png",
                mime="image/png",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
