
"""
Evaluate leave-one-out reconstruction performance of the baseline symptom space.

This script performs leave-one-out cross-validation on the sporadic baseline
MDS-UPDRS dataset. For each held-out patient, a low-dimensional symptom space
is estimated from the remaining patients using Spectral Estimation.

The held-out patient is projected into the estimated space and reconstructed.
Reconstruction performance is compared against a shuffled eigenvector-space
baseline using:

    1. Root mean squared error (RMSE)
    2. Correlation between original and reconstructed symptom profiles

Outputs
-------
saved_data/
    Data_Leave_one_out_<class_name>_BL_and_random_space.pkl

figures/
    RMSE_robustness_<class_name>.svg
    RMSE_robustness_<class_name>.png
    Corr_robustness_<class_name>.svg
    Corr_robustness_<class_name>.png
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import __init__ as nnr  # Network Noise Rejection functions
import data_formatting_functions as data_fun
from final_figure_config import fig_properties, apply_figure_properties
import spectral_estimation as se

# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


# ---------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------

def compute_samplewise_correlation(original, reconstructed):
    """
    Compute sample-wise Pearson correlations between original and reconstructed data.

    Parameters
    ----------
    original : np.ndarray
        Original data with shape (n_samples, n_features).

    reconstructed : np.ndarray
        Reconstructed data with shape (n_samples, n_features).

    Returns
    -------
    mean_corr : float
        Mean sample-wise Pearson correlation.

    std_corr : float
        Standard deviation of sample-wise Pearson correlations.

    all_corrs : np.ndarray
        Pearson correlation for each sample.
    """

    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed arrays must have the same shape."
        )

    all_corrs = np.array(
        [
            np.corrcoef(original_sample, reconstructed_sample)[0, 1]
            for original_sample, reconstructed_sample in zip(
                original,
                reconstructed,
            )
        ]
    )

    return np.mean(all_corrs), np.std(all_corrs), all_corrs


# ---------------------------------------------------------------------
# Load baseline data
# ---------------------------------------------------------------------

class_name = "sporadic"

with open(
    os.path.join(saved_data_dir, f"data_baseline_{class_name}.pkl"),
    "rb",
) as file:
    loaded_data = pickle.load(file)

dataframes_rank_norm = loaded_data["dataframes_BL_rank_norm"]
test_names = loaded_data["test_names"]
BL_patients_list = loaded_data["BL_patients_list"]


# ---------------------------------------------------------------------
# Initialise storage variables
# ---------------------------------------------------------------------

loop_test_rmse_SE = []
loop_train_rmse_SE = []

loop_corr_SE_train = []
loop_corr_SE_test = []

random_loop_test_rmse_SE = []
random_loop_train_rmse_SE = []

random_loop_corr_SE_train = []
random_loop_corr_SE_test = []

total_dims = []


# ---------------------------------------------------------------------
# Leave-one-out reconstruction analysis
# ---------------------------------------------------------------------

runs = 1

for loop in range(1, runs + 1):

    n = dataframes_rank_norm.shape[0]

    for i in range(0, n):

        test_index = np.array(i)
        train_indices = np.array(
            [
                j
                for j in range(0, n)
                if j != i
            ]
        )

        # Define leave-one-out train and test sets.
        train_set = dataframes_rank_norm.iloc[train_indices]
        test_set = dataframes_rank_norm.iloc[test_index]

        # Spectral-estimation space from the leave-one-out training set.
        try:
            space = se.build_spectral_space(train_set)
        except se.FeatureDroppedError:
            print("Skipped: a feature dropped from the network.")
            continue

        exceeding_eig_space = space.exceeding_space
        exceeding_space_dims = space.exceeding_dims

        test_array = np.asarray(test_set)
        train_array = np.asarray(train_set)

        # Create a shuffled low-dimensional space as a random baseline.
        shuffled_eig_space = exceeding_eig_space[
            np.random.permutation(len(exceeding_eig_space))
        ]

        # Reconstruction using the true Spectral Estimation space.
        SE_rmse_test, data_reconstructed_test, projected_data_test = (
            data_fun.project_and_recover(test_array, exceeding_eig_space)
        )

        SE_rmse_train, data_reconstructed_train, projected_data_train = (
            data_fun.project_and_recover(train_array, exceeding_eig_space)
        )

        # Reconstruction using the shuffled comparison space.
        random_SE_rmse_test, random_data_reconstructed_test, random_projected_data_test = (
            data_fun.project_and_recover(test_array, shuffled_eig_space)
        )

        random_SE_rmse_train, random_data_reconstructed_train, random_projected_data_train = (
            data_fun.project_and_recover(train_array, shuffled_eig_space)
        )

        loop_test_rmse_SE.append(SE_rmse_test)
        loop_train_rmse_SE.append(SE_rmse_train)

        random_loop_test_rmse_SE.append(random_SE_rmse_test)
        random_loop_train_rmse_SE.append(random_SE_rmse_train)

        total_dims.append(exceeding_space_dims)

        # Correlation for training data using the true space.
        mean_train_corr, std_train_corr, train_corr = (
            compute_samplewise_correlation(
                train_array,
                data_reconstructed_train,
            )
        )

        loop_corr_SE_train.append(mean_train_corr)

        # Correlation for held-out test patient using the true space.
        loop_corr_SE_test.append(
            np.corrcoef(data_reconstructed_test, test_array)[0, 1]
        )

        # Correlation for training data using the shuffled space.
        random_mean_train_corr, random_std_train_corr, random_train_corr = (
            compute_samplewise_correlation(
                train_array,
                random_data_reconstructed_train,
            )
        )

        random_loop_corr_SE_train.append(random_mean_train_corr)

        # Correlation for held-out test patient using the shuffled space.
        random_loop_corr_SE_test.append(
            np.corrcoef(random_data_reconstructed_test, test_array)[0, 1]
        )


# ---------------------------------------------------------------------
# Convert results to arrays and dataframes
# ---------------------------------------------------------------------

test_rmse_SE_array = np.array(loop_test_rmse_SE)
train_rmse_SE_array = np.array(loop_train_rmse_SE)

random_test_rmse_SE_array = np.array(random_loop_test_rmse_SE)
random_train_rmse_SE_array = np.array(random_loop_train_rmse_SE)

total_dims_array = np.array(total_dims)

test_corr_SE_array = np.array(loop_corr_SE_test)
train_corr_SE_array = np.array(loop_corr_SE_train)

random_test_corr_SE_array = np.array(random_loop_corr_SE_test)
random_train_corr_SE_array = np.array(random_loop_corr_SE_train)

df_rmse = pd.DataFrame(
    {
        "RMSE": test_rmse_SE_array,
        "RMSE:train SE": train_rmse_SE_array,
        "RMSE:random test SE": random_test_rmse_SE_array,
        "RMSE:random train SE": random_train_rmse_SE_array,
    }
)

df_corr = pd.DataFrame(
    {
        "Correlation": test_corr_SE_array,
        "Corr:train SE": train_corr_SE_array,
        "Corr:random test SE": random_test_corr_SE_array,
        "Corr:random train SE": random_train_corr_SE_array,
    }
)

df_performance_RMSE = pd.DataFrame(
    {
        "RMSE:test SE": test_rmse_SE_array,
        "RMSE:train SE": train_rmse_SE_array,
        "shuffled RMSE:test SE": random_test_rmse_SE_array,
        "shuffled RMSE:train SE": random_train_rmse_SE_array,
    }
)

df_performance_corr = pd.DataFrame(
    {
        "Corr:test SE": test_corr_SE_array,
        "Corr:train SE": train_corr_SE_array,
        "shuffled Corr:test SE": random_test_corr_SE_array,
        "shuffled Corr:train SE": random_train_corr_SE_array,
    }
)


# ---------------------------------------------------------------------
# Save leave-one-out performance results
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    f"Data_Leave_one_out_{class_name}_BL_and_random_space.pkl",
)

pickle_data = {
    "df_rmse": df_rmse,
    "df_corr": df_corr,
    "total_dims": total_dims_array,
    "test_names": test_names,
    "BL_patients_list": BL_patients_list,
    "df_performance_RMSE": df_performance_RMSE,
    "df_performance_corr": df_performance_corr,
}

with open(output_file, "wb") as file:
    pd.to_pickle(pickle_data, file)

print("Saved leave-one-out reconstruction results to:")
print(output_file)


# ---------------------------------------------------------------------
# Plot reconstruction RMSE
# ---------------------------------------------------------------------

fig = plt.figure(figsize=fig_properties["figsize"]["violin"])

ax_rmse = sns.violinplot(
    data=test_rmse_SE_array,
    palette=["blue"],
)

plt.ylabel("RMSE", fontsize=fig_properties["fontsize"])
plt.xticks([])
plt.yticks(fontsize=fig_properties["fontsize"])
plt.ylim(0, 1)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

apply_figure_properties(fig, fig_properties)

plt.savefig(
    os.path.join(figures_dir, f"RMSE_robustness_{class_name}.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
)

plt.savefig(
    os.path.join(figures_dir, f"RMSE_robustness_{class_name}.png"),
    format="png",
    dpi=fig_properties["dpi"],
)

plt.show(block=False)


# ---------------------------------------------------------------------
# Plot reconstruction correlation
# ---------------------------------------------------------------------

plt.figure(figsize=fig_properties["figsize"]["violin"])

ax_corr = sns.violinplot(
    data=test_corr_SE_array,
    palette=["salmon"],
)

plt.ylabel("Correlation", fontsize=fig_properties["fontsize"])
plt.xticks([])
plt.yticks(fontsize=fig_properties["fontsize"])
plt.ylim(0, 1)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

plt.savefig(
    os.path.join(figures_dir, f"Corr_robustness_{class_name}.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
)

plt.savefig(
    os.path.join(figures_dir, f"Corr_robustness_{class_name}.png"),
    format="png",
    dpi=fig_properties["dpi"],
)

plt.show(block=False)


plt.show()
