
"""
Plot stability of the baseline low-dimensional space across sample sizes.

This script loads precomputed subset-stability results and plots how the
baseline low-dimensional symptom space changes with sample size.

The following outputs are generated:
    1. Weighted cosine similarity and subspace dimension on twin y-axes.
    2. Subspace dimension only.
    3. Weighted cosine similarity only.
    4. Median subspace dimension.

Outputs
-------
figures/
    N_fold_BL_spaces.svg
    N_fold_BL_spaces.png
    N_fold_BL_spaces_dimension_only.svg
    N_fold_BL_spaces_dimension_only.png
    N_fold_BL_spaces_cos_simi_only.svg
    N_fold_BL_spaces_cos_simi_only.png
    N_fold_BL_spaces_dimension_median.svg
    N_fold_BL_spaces_dimension_median.png
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from final_figure_config import fig_properties, apply_figure_properties


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


# ---------------------------------------------------------------------
# Load subset-stability results
# ---------------------------------------------------------------------

input_file = os.path.join(
    saved_data_dir,
    "Data_Manuscript_data_weighted_cosine_BL_spaces_N_subsets_final.pkl",
)

with open(input_file, "rb") as file:
    loaded_data = pickle.load(file)

cos_similarity_N = loaded_data["cos_similarity_N"]
excess_dims_N = loaded_data["excess_dims_N"]
N_values = loaded_data["N_values"]
num_repeats = loaded_data["num_repeats"]


# ---------------------------------------------------------------------
# Compute summary statistics
# ---------------------------------------------------------------------

mean_cos_similarity = {
    N: np.mean(cos_similarity_N[N])
    for N in N_values
}

std_err_cos_similarity = {
    N: np.std(cos_similarity_N[N]) / np.sqrt(num_repeats)
    for N in N_values
}

mean_excess_dims = {
    N: np.mean(excess_dims_N[N])
    for N in N_values
}

std_err_excess_dims = {
    N: np.std(excess_dims_N[N]) / np.sqrt(num_repeats)
    for N in N_values
}

x = np.array(N_values)


# ---------------------------------------------------------------------
# Plot weighted cosine similarity and subspace dimension together
# ---------------------------------------------------------------------

fig, ax1 = plt.subplots(
    figsize=(12 / 2.5, 8 / 2.5),
    dpi=300,
)

ax1.errorbar(
    x,
    [mean_cos_similarity[N] for N in N_values],
    yerr=[std_err_cos_similarity[N] for N in N_values],
    fmt="-o",
    color="b",
    label="Weighted Cosine Similarity at different N's of baseline",
)

ax1.set_xlabel("N", fontsize=fig_properties["fontsize"])
ax1.set_ylim(0, 1.2)

ax1.set_ylabel(
    "Weighted Cosine Similarity",
    color="b",
    fontsize=fig_properties["fontsize"],
)

ax1.tick_params(axis="y", labelcolor="b")

lines_1, labels_1 = ax1.get_legend_handles_labels()

ax2 = ax1.twinx()

ax2.errorbar(
    x,
    [mean_excess_dims[N] for N in N_values],
    yerr=[std_err_excess_dims[N] for N in N_values],
    fmt="-s",
    color="r",
    label="Dimension of the Subspace",
)

ax2.set_ylabel(
    "Dimension of the Subspace",
    color="r",
    fontsize=fig_properties["fontsize"],
)

ax2.tick_params(axis="y", labelcolor="r")

lines_2, labels_2 = ax2.get_legend_handles_labels()

fig.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="upper left",
    fontsize=fig_properties["labelsize"],
)

apply_figure_properties(fig, fig_properties)

fig.tight_layout()

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
)

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces.png"),
    format="png",
    dpi=fig_properties["dpi"],
)

plt.show(block=False)


# ---------------------------------------------------------------------
# Plot subspace dimension only
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12 / 2.5, 8 / 2.5),
    dpi=300,
)

ax.errorbar(
    x,
    [mean_excess_dims[N] for N in N_values],
    yerr=[std_err_excess_dims[N] for N in N_values],
    fmt="-s",
    color="r",
    label="Dimension of the Subspace",
)

ax.set_xlabel("N", fontsize=fig_properties["fontsize"])

ax.set_ylabel(
    "Dimension of the Subspace",
    fontsize=fig_properties["fontsize"],
    color="r",
)

apply_figure_properties(fig, fig_properties)

fig.tight_layout()

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces_dimension_only.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
)

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces_dimension_only.png"),
    format="png",
    dpi=fig_properties["dpi"],
)

plt.show(block=False)


# ---------------------------------------------------------------------
# Plot weighted cosine similarity only
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12 / 2.5, 8 / 2.5),
    dpi=300,
)

ax.errorbar(
    x,
    [mean_cos_similarity[N] for N in N_values],
    yerr=[std_err_cos_similarity[N] for N in N_values],
    fmt="-o",
    color="b",
    label="Weighted Cosine Similarity at different N's of baseline",
)

ax.set_xlabel("N", fontsize=fig_properties["fontsize"])
ax.set_ylim(0, 1)

ax.set_ylabel(
    "Weighted Cosine Similarity",
    color="b",
    fontsize=fig_properties["fontsize"],
)

ax.tick_params(axis="y", labelcolor="b")

apply_figure_properties(fig, fig_properties)

fig.tight_layout()

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces_cos_simi_only.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
)

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces_cos_simi_only.png"),
    format="png",
    dpi=fig_properties["dpi"],
)

plt.show(block=False)


# ---------------------------------------------------------------------
# Plot median subspace dimension
# ---------------------------------------------------------------------

median_excess_dims = {
    N: np.median(excess_dims_N[N])
    for N in N_values
}

# Approximate standard error of the median under normality.
se_median_excess_dims = {
    N: np.sqrt(np.pi / 2)
    * np.std(excess_dims_N[N])
    / np.sqrt(len(excess_dims_N[N]))
    for N in N_values
}

fig, ax = plt.subplots(
    figsize=(12 / 2.5, 8 / 2.5),
    dpi=300,
)

ax.errorbar(
    x,
    [median_excess_dims[N] for N in N_values],
    yerr=[se_median_excess_dims[N] for N in N_values],
    fmt="-s",
    color="r",
    label="Median Dimension of the Subspace",
)

ax.set_xlabel("N", fontsize=fig_properties["fontsize"])

ax.set_ylabel(
    "Dimension of the Subspace",
    fontsize=fig_properties["fontsize"],
    color="r",
)

apply_figure_properties(fig, fig_properties)

max_y = max(
    median_excess_dims[N] + se_median_excess_dims[N]
    for N in N_values
)

upper_limit = np.ceil(max_y / 3) * 3

ax.set_ylim(0, upper_limit)
ax.set_yticks(np.arange(0, upper_limit + 1, 3))

fig.tight_layout()

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces_dimension_median.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
)

plt.savefig(
    os.path.join(figures_dir, "N_fold_BL_spaces_dimension_median.png"),
    format="png",
    dpi=fig_properties["dpi"],
)

plt.show(block=False)


plt.show()
