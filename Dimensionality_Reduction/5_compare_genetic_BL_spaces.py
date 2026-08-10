
"""
Compare low-dimensional baseline spaces from genetic and sporadic cohorts.

This script reconstructs the low-dimensional symptom space for the genetic
baseline cohort using Spectral Estimation, then compares it with the
previously saved sporadic baseline space.

Subspace similarity is quantified using principal angles between the two
spaces. The cosine similarities are weighted by the genetic-space eigenvalues
to obtain a weighted cosine similarity score.

Output
------
saved_data/
    Weighted_cosine_BL_Genetic.pkl
"""

import os
import pickle

import numpy as np
from scipy.linalg import svd

import __init__ as nnr  # Network Noise Rejection functions
import data_formatting_functions as data_fun


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")


# ---------------------------------------------------------------------
# Load genetic baseline data
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_genetic.pkl"),
    "rb",
) as file:
    genetic_data = pickle.load(file)

dataframes_rank_norm = genetic_data["dataframes_rank_norm"]

train_set = dataframes_rank_norm
genetic_N = dataframes_rank_norm.shape[0]


# ---------------------------------------------------------------------
# Recompute genetic low-dimensional space
# ---------------------------------------------------------------------

import spectral_estimation as se

space = se.build_spectral_space(train_set)

exceeding_eig_space_genetic  = space.exceeding_space
exceeding_space_dims_genetic = space.exceeding_dims
exceeding_eig_vals           = space.exceeding_eig_vals

# ---------------------------------------------------------------------
# Load previously saved sporadic baseline space
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_sporadic.pkl"),
    "rb",
) as file:
    sporadic_data = pickle.load(file)

exceeding_eig_space = sporadic_data["exceeding_eig_space"]


# ---------------------------------------------------------------------
# Compare sporadic and genetic low-dimensional spaces
# ---------------------------------------------------------------------

# Reverse eigenvector order to match the original principal-component ordering.
U1 = exceeding_eig_space[:, ::-1]
U2 = exceeding_eig_space_genetic[:, ::-1]

eig_vals = exceeding_eig_vals

# Normalize genetic-space eigenvalues to use as weights.
eig_weights = eig_vals / np.sum(eig_vals)

# Dot product between the two subspaces.
M = np.dot(U1.T, U2)

# Singular values of the cross-space dot-product matrix define the
# cosines of the principal angles between subspaces.
_, singular_values, _ = svd(M)

angles = np.arccos(np.clip(singular_values, -1.0, 1.0))
cos_principal_angles = np.cos(angles)

# Weighted mean cosine similarity between the two spaces.
weighted_cosine_similarity = np.sum(
    cos_principal_angles * eig_weights[:len(cos_principal_angles)]
)

print(
    f"weighted_cosine_similarity: {weighted_cosine_similarity}, "
    f"for a sample size of {genetic_N} and "
    f"{exceeding_space_dims_genetic} exceeding dimensions"
)


# ---------------------------------------------------------------------
# Save similarity result
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    "Weighted_cosine_BL_Genetic.pkl",
)

with open(output_file, "wb") as file:
    pickle.dump(
        {
            "cos_similarityBLvsGenetic": weighted_cosine_similarity,
            "exceeding_space_dims_genetic": exceeding_space_dims_genetic,
            "genetic_N": genetic_N,
        },
        file,
    )

print("Saved weighted cosine similarity result to:")
print(output_file)

