
"""
Evaluate how well the sporadic baseline low-dimensional space generalizes
to the genetic cohort.

This script loads the low-dimensional symptom space estimated from the
sporadic baseline cohort and uses it to reconstruct:

    1. Sporadic baseline data
    2. Genetic baseline data

Reconstruction quality is quantified using:
    - patient-wise correlation between original and reconstructed scores
    - patient-wise root mean squared error (RMSE)

The two cohorts are compared using Mann-Whitney U tests, and violin plots
are generated for both reconstruction correlation and RMSE.
"""

import os
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import data_formatting_functions as data_fun
from final_figure_config import fig_properties


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_corr_SE(input_array, eig_space):
    """
    Compute patient-wise reconstruction correlation.

    Parameters
    ----------
    input_array : np.ndarray
        Original patient-by-feature data matrix.

    eig_space : np.ndarray
        Low-dimensional eigenvector space used for projection and recovery.

    Returns
    -------
    np.ndarray
        Correlation between original and reconstructed feature profiles
        for each patient.
    """

    _, data_reconstructed, _ = data_fun.project_and_recover(
        input_array,
        eig_space
    )

    corr = np.array([
        np.corrcoef(data_reconstructed[row], input_array[row])[0, 1]
        for row in range(len(input_array))
    ])

    return corr


def get_rmse(input_array, eig_space):
    """
    Compute patient-wise reconstruction error.

    Parameters
    ----------
    input_array : np.ndarray
        Original patient-by-feature data matrix.

    eig_space : np.ndarray
        Low-dimensional eigenvector space used for projection and recovery.

    Returns
    -------
    np.ndarray
        Root mean squared error between original and reconstructed feature
        profiles for each patient.
    """

    _, data_reconstructed, _ = data_fun.project_and_recover(
        input_array,
        eig_space
    )

    rmse = np.array([
        np.sqrt(np.mean((data_reconstructed[row] - input_array[row]) ** 2))
        for row in range(len(input_array))
    ])

    return rmse


# ---------------------------------------------------------------------
# Load sporadic baseline low-dimensional space
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_sporadic.pkl"),
    "rb"
) as file:
    data_BL = pickle.load(file)

BL_data = data_BL["dataframes_rank_norm"]
exceeding_eig_space = data_BL["exceeding_eig_space"]

BL_data_array = np.asarray(BL_data)


# ---------------------------------------------------------------------
# Load genetic baseline data
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_baseline_genetic.pkl"),
    "rb"
) as file:
    data_gen = pickle.load(file)

gen_array = np.asarray(data_gen["dataframes_BL_rank_norm"])


# ---------------------------------------------------------------------
# Compute reconstruction quality
# ---------------------------------------------------------------------

# Reconstruction quality for genetic baseline data using the sporadic
# baseline low-dimensional space.
gen_corr_SE = get_corr_SE(gen_array, exceeding_eig_space)
gen_rmse = get_rmse(gen_array, exceeding_eig_space)

# Reconstruction quality for the original sporadic baseline data.
BL_corr_SE = get_corr_SE(BL_data_array, exceeding_eig_space)
rmse_BL = get_rmse(BL_data_array, exceeding_eig_space)


# ---------------------------------------------------------------------
# Statistical comparison between cohorts
# ---------------------------------------------------------------------

u_stat_corr, p_val_corr = mannwhitneyu(
    gen_corr_SE,
    BL_corr_SE,
    alternative="two-sided"
)

print(
    f"Mann-Whitney U (Correlation): "
    f"U = {u_stat_corr:.3f}, p = {p_val_corr:.3e}"
)

u_stat_rmse, p_val_rmse = mannwhitneyu(
    gen_rmse,
    rmse_BL,
    alternative="two-sided"
)

print(
    f"Mann-Whitney U (RMSE): "
    f"U = {u_stat_rmse:.3f}, p = {p_val_rmse:.3e}"
)


# ---------------------------------------------------------------------
# Prepare data for plotting
# ---------------------------------------------------------------------

all_corrs = np.concatenate([gen_corr_SE, BL_corr_SE])
all_rmse = np.concatenate([gen_rmse, rmse_BL])

labels = (
    ["Genetic Data"] * len(gen_corr_SE)
    + ["BL data"] * len(BL_corr_SE)
)

bl_median_corr = np.median(BL_corr_SE)
bl_median_rmse = np.median(rmse_BL)


# ---------------------------------------------------------------------
# Plot reconstruction correlation
# ---------------------------------------------------------------------

plt.figure(figsize=(12 / 2.54, 10 / 2.54), dpi=300)

sns.violinplot(
    x=labels,
    y=all_corrs,
    inner="box"
)

plt.axhline(
    bl_median_corr,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"BL median: {bl_median_corr:.2f}"
)

plt.legend(fontsize=12)
plt.ylabel("Correlation (Original vs. Reconstructed)", fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

ax = plt.gca()

plt.text(
    0.5,
    0.95,
    f"p = {p_val_corr:.3e}",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=11
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

plt.savefig(
    os.path.join(figures_dir, "how_good_is_BL_space_on_genetic_cohort.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
    transparent=True
)

plt.show(block=False)


# ---------------------------------------------------------------------
# Plot reconstruction RMSE
# ---------------------------------------------------------------------

plt.figure(figsize=(12 / 2.54, 10 / 2.54), dpi=300)

sns.violinplot(
    x=labels,
    y=all_rmse,
    inner="box"
)

plt.axhline(
    bl_median_rmse,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"BL median: {bl_median_rmse:.2f}"
)

plt.legend(fontsize=12)
plt.ylabel("RMSE (Original vs. Reconstructed)", fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

ax = plt.gca()

plt.text(
    0.5,
    0.95,
    f"p = {p_val_rmse:.3e}",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=11
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

plt.savefig(
    os.path.join(figures_dir, "how_good_is_BL_space_on_genetic_cohort_RMSE.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
    transparent=True
)

plt.show(block=False)



plt.show()
