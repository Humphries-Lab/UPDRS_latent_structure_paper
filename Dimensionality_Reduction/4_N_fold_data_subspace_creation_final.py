
"""
Assess stability of the baseline low-dimensional space across sample sizes.

This script repeatedly samples subsets of the sporadic baseline cohort,
recomputes the low-dimensional symptom space for each subset using Network
Noise Rejection, and compares each subset space with the full baseline
space using weighted cosine similarity based on principal angles.

Eigenvectors and eigenvalues are reordered when needed so that dimensions
are compared in descending eigenvalue order.

For each subset size, the script stores:
    - weighted cosine similarity to the full baseline space
    - number of dimensions exceeding the null expectation

Output
------
saved_data/
    Data_Manuscript_data_weighted_cosine_BL_spaces_N_subsets_final.pkl
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

import __init__ as nnr  # Network Noise Rejection functions
import data_formatting_functions as data_fun
import spectral_estimation as se


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")


# ---------------------------------------------------------------------
# Load full baseline low-dimensional space
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_sporadic.pkl"),
    "rb",
) as file:
    loaded_data = pickle.load(file)

dataframes_rank_norm = loaded_data["dataframes_rank_norm"]

U1 = loaded_data["exceeding_eig_space"].copy()
eig_vals_U1 = loaded_data["eigen_vals"].copy()


# ---------------------------------------------------------------------
# Define subset sizes and repetitions
# ---------------------------------------------------------------------

N_values = [100, 200, 400, 600, 700, 800]
total_data_points = dataframes_rank_norm.shape[0]

num_repeats = 50


# ---------------------------------------------------------------------
# Prepare full-space eigenvector weights
# ---------------------------------------------------------------------

# Detect whether eigenvalues are in ascending order.
# If so, reverse eigenvalues and eigenvectors so that dimensions are ordered
# from largest to smallest eigenvalue.
if eig_vals_U1[0] < eig_vals_U1[-1]:
    eig_vals_U1 = eig_vals_U1[::-1]
    U1 = U1[:, ::-1]

# Normalize full-space eigenvalues to use as dimension weights.
eig_weights_full = eig_vals_U1 / np.sum(eig_vals_U1)

assert U1.shape[1] == eig_weights_full.shape[0], (
    "Mismatch between U1 columns and eig_weights."
)


# ---------------------------------------------------------------------
# Recompute low-dimensional spaces for random subsets
# ---------------------------------------------------------------------

results = {}
results_flipped = {}
cos_similarity_N = {}
excess_dims_N = {}

for N in N_values:

    cos_simi_loop = []
    results[N] = []

    cos_simi_loop_flipped = []
    results_flipped[N] = []

    excess_dim_loop = []

    for _ in range(num_repeats):

        if N == total_data_points:
            train_set = dataframes_rank_norm
        else:
            train_set = dataframes_rank_norm.sample(
                n=N,
                replace=False,
            )

        correlation_matrix = train_set.corr(method="pearson")

        if correlation_matrix.isna().any().any():
            print("NaNs found in the correlation matrix. Skipping this iteration.")
            continue

        # Spectral-estimation space from this subset.
        try:
            space = se.build_spectral_space(train_set)
        except se.FeatureDroppedError:
            print("Skipped subset: a feature dropped from the network.")
            continue

        exceeding_eig_space = space.exceeding_space
        exceeding_space_dims = space.exceeding_dims
        exceeding_eig_vals = space.exceeding_eig_vals

        excess_dim_loop.append(exceeding_space_dims)

        U2 = exceeding_eig_space.copy()
        eig_vals_U2 = exceeding_eig_vals.copy()

        assert U2.shape[0] == U1.shape[0], (
            "Ambient dimension mismatch between U1 and U2."
        )

        # Match subset-space ordering to the full-space ordering.
        if eig_vals_U2[0] < eig_vals_U2[-1]:
            U2 = U2[:, ::-1]
            eig_vals_U2 = eig_vals_U2[::-1]

        # Singular values of U1.T @ U2 are the cosines of the principal angles.
        M = U1.T @ U2
        _, singular_values, _ = svd(M, full_matrices=False)

        k = min(U1.shape[1], U2.shape[1])
        sv = np.clip(singular_values[:k], -1.0, 1.0)

        # Use full-space eigenvalue weights, truncated to the common number
        # of dimensions and renormalized.
        weights = eig_weights_full[:k].copy()

        if np.sum(weights) <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / np.sum(weights)

        weighted_cosine_similarity = float(np.sum(weights * sv))

        cos_simi_loop.append(weighted_cosine_similarity)
        results[N].append(weighted_cosine_similarity)

    excess_dims_N[N] = excess_dim_loop
    cos_similarity_N[N] = cos_simi_loop


# ---------------------------------------------------------------------
# Summarize subset stability results
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


# ---------------------------------------------------------------------
# Plot stability across subset sizes
# ---------------------------------------------------------------------

x = np.array(N_values)

fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.errorbar(
    x,
    [mean_cos_similarity[N] for N in N_values],
    yerr=[std_err_cos_similarity[N] for N in N_values],
    fmt="-o",
    color="b",
    label="Weighted Cosine Similarity",
)

ax1.set_xlabel("N", fontsize=14, fontweight="bold")
ax1.set_ylabel(
    "Weighted Cosine Similarity",
    color="b",
    fontsize=14,
    fontweight="bold",
)

ax1.tick_params(axis="y", labelcolor="b", labelsize=12)
ax1.tick_params(axis="x", labelsize=12)

ax2 = ax1.twinx()

ax2.errorbar(
    x,
    [mean_excess_dims[N] for N in N_values],
    yerr=[std_err_excess_dims[N] for N in N_values],
    fmt="-s",
    color="r",
    label="Excess Dimensions",
)

ax2.set_ylabel(
    "Excess Dimensions",
    color="r",
    fontsize=14,
    fontweight="bold",
)

ax2.tick_params(axis="y", labelcolor="r", labelsize=12)

plt.title(
    "Weighted Cosine Similarity and Excess Dimensions Across N",
    fontsize=16,
    fontweight="bold",
)

fig1.tight_layout()


# ---------------------------------------------------------------------
# Save subset stability results
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    "Data_Manuscript_data_weighted_cosine_BL_spaces_N_subsets_final.pkl",
)

with open(output_file, "wb") as file:
    pickle.dump(
        {
            "cos_similarity_N": cos_similarity_N,
            "excess_dims_N": excess_dims_N,
            "N_values": N_values,
            "num_repeats": num_repeats,
        },
        file,
    )

print("Saved subset stability results to:")
print(output_file)

plt.show(block=False)



plt.show()
