
"""
Evaluate train-test reconstruction performance of the baseline symptom space.

This script performs repeated 5-fold cross-validation on the sporadic baseline
MDS-UPDRS dataset. For each train/test split, a low-dimensional symptom space
is estimated from the training data using Network Noise Rejection.

Reconstruction performance is evaluated for:

    1. Spectral Estimation (SE) space
    2. EVD space with the same number of dimensions as the SE space
    3. Shuffled SE space as a random-control comparison

Performance is quantified using:
    - RMSE
    - correlation between original and reconstructed symptom profiles

Outputs
-------
saved_data/
    Data_test_train_<class_name>_BL_<nfold>fold_<runs>runs.pkl

figures/
    RMSE_robustness_<class_name>.svg
    RMSE_robustness_<class_name>.png
    Corr_robustness_<class_name>.svg
    Corr_robustness_<class_name>.png
"""

import os
import pickle
import spectral_estimation as se
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold

import __init__ as nnr  # Network Noise Rejection functions
import data_formatting_functions as data_fun
from final_figure_config import fig_properties, apply_figure_properties


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


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
# Cross-validation settings
# ---------------------------------------------------------------------

nfold = 5
runs = 50


# ---------------------------------------------------------------------
# Initialise storage variables
# ---------------------------------------------------------------------

loop_test_rmse_SE = []
loop_train_rmse_SE = []
loop_test_rmse_EVD = []
loop_train_rmse_EVD = []

loop_corr_SE_train = []
loop_corr_EVD_train = []
loop_corr_SE_test = []
loop_corr_EVD_test = []

shuffled_loop_test_rmse_SE = []
shuffled_loop_train_rmse_SE = []
shuffled_loop_test_corr_SE = []
shuffled_loop_train_corr_SE = []

rel_loss_test = []
rel_loss_train = []
rel_corr_test = []
rel_corr_train = []

total_dims = []
exitcount = 0


# ---------------------------------------------------------------------
# Repeated train-test reconstruction analysis
# ---------------------------------------------------------------------

for loop in range(1, runs + 1):

    kf = KFold(n_splits=nfold, shuffle=True)

    for _, (train_index, test_index) in enumerate(kf.split(dataframes_rank_norm)):

        train_set = dataframes_rank_norm.iloc[train_index]
        test_set = dataframes_rank_norm.iloc[test_index]

        # EVD of the raw correlation matrix — kept as the comparison baseline.
        correlation_matrix = train_set.corr(method="pearson")
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Spectral-estimation space from the training fold.
        try:
            space = se.build_spectral_space(train_set)
        except se.FeatureDroppedError:
            print("Skipped fold: a feature dropped from the network.")
            exitcount += 1
            continue

        exceeding_eig_space = space.exceeding_space
        exceeding_space_dims = space.exceeding_dims

        # Shuffled SE space as a random-control comparison.
        shuffled_eig_space = exceeding_eig_space[
            np.random.permutation(len(exceeding_eig_space))
        ]

        test_array = np.asarray(test_set)
        train_array = np.asarray(train_set)


        # -------------------------------------------------------------
        # Reconstruction using SE space
        # -------------------------------------------------------------

        SE_rmse_test, data_reconstructed_test, projected_data_test = (
            data_fun.project_and_recover(test_array, exceeding_eig_space)
        )

        SE_rmse_train, data_reconstructed_train, projected_data_train = (
            data_fun.project_and_recover(train_array, exceeding_eig_space)
        )


        # -------------------------------------------------------------
        # Reconstruction using shuffled SE space
        # -------------------------------------------------------------

        shuffled_SE_rmse_test, shuffled_data_reconstructed_test, shuffled_projected_data_test = (
            data_fun.project_and_recover(test_array, shuffled_eig_space)
        )

        shuffled_SE_rmse_train, shuffled_data_reconstructed_train, shuffled_projected_data_train = (
            data_fun.project_and_recover(train_array, shuffled_eig_space)
        )

        loop_test_rmse_SE.append(SE_rmse_test)
        loop_train_rmse_SE.append(SE_rmse_train)

        shuffled_loop_test_rmse_SE.append(shuffled_SE_rmse_test)
        shuffled_loop_train_rmse_SE.append(shuffled_SE_rmse_train)

        total_dims.append(exceeding_space_dims)


        # -------------------------------------------------------------
        # Correlation for SE and shuffled SE reconstructions
        # -------------------------------------------------------------

        train_corr_SE = np.zeros((train_array.shape[0],))
        test_corr_SE = np.zeros((test_array.shape[0],))

        shuffled_train_corr_SE = np.zeros((train_array.shape[0],))
        shuffled_test_corr_SE = np.zeros((test_array.shape[0],))

        for row in range(test_array.shape[0]):
            test_corr_SE[row] = np.corrcoef(
                data_reconstructed_test[row],
                test_array[row],
            )[0, 1]

            shuffled_test_corr_SE[row] = np.corrcoef(
                shuffled_data_reconstructed_test[row],
                test_array[row],
            )[0, 1]

        for row in range(train_array.shape[0]):
            train_corr_SE[row] = np.corrcoef(
                data_reconstructed_train[row],
                train_array[row],
            )[0, 1]

            shuffled_train_corr_SE[row] = np.corrcoef(
                shuffled_data_reconstructed_train[row],
                train_array[row],
            )[0, 1]

        loop_corr_SE_test.append(np.mean(test_corr_SE))
        loop_corr_SE_train.append(np.mean(train_corr_SE))

        shuffled_loop_test_corr_SE.append(np.mean(shuffled_test_corr_SE))
        shuffled_loop_train_corr_SE.append(np.mean(shuffled_train_corr_SE))


        # -------------------------------------------------------------
        # Reconstruction using EVD space
        # -------------------------------------------------------------

        selected_eigenvalues = eigenvalues[:exceeding_space_dims]
        selected_eigenvectors = eigenvectors[:, :exceeding_space_dims]

        EVD_rmse_test, data_reconstructed_test, projected_data_test = (
            data_fun.project_and_recover(test_array, selected_eigenvectors)
        )

        EVD_rmse_train, data_reconstructed_train, projected_data_train = (
            data_fun.project_and_recover(train_array, selected_eigenvectors)
        )

        loop_test_rmse_EVD.append(EVD_rmse_test)
        loop_train_rmse_EVD.append(EVD_rmse_train)

        train_corr_evd = np.zeros((train_array.shape[0],))
        test_corr_evd = np.zeros((test_array.shape[0],))

        for row in range(test_array.shape[0]):
            test_corr_evd[row] = np.corrcoef(
                data_reconstructed_test[row],
                test_array[row],
            )[0, 1]

        for row in range(train_array.shape[0]):
            train_corr_evd[row] = np.corrcoef(
                data_reconstructed_train[row],
                train_array[row],
            )[0, 1]

        loop_corr_EVD_test.append(np.mean(test_corr_evd))
        loop_corr_EVD_train.append(np.mean(train_corr_evd))


        # -------------------------------------------------------------
        # Relative SE vs EVD performance
        # -------------------------------------------------------------

        relative_se_evd_loss_test = (
            SE_rmse_test - EVD_rmse_test
        ) / SE_rmse_test

        relative_se_evd_loss_train = (
            SE_rmse_train - EVD_rmse_train
        ) / SE_rmse_train

        relative_se_evd_corr_test = (
            np.mean(test_corr_evd) - np.mean(test_corr_SE)
        ) / np.mean(test_corr_SE)

        relative_se_evd_corr_train = (
            np.mean(train_corr_evd) - np.mean(train_corr_SE)
        ) / np.mean(train_corr_SE)

        rel_loss_test.append(relative_se_evd_loss_test)
        rel_loss_train.append(relative_se_evd_loss_train)

        rel_corr_test.append(relative_se_evd_corr_test)
        rel_corr_train.append(relative_se_evd_corr_train)


# ---------------------------------------------------------------------
# Convert results to arrays
# ---------------------------------------------------------------------

total_dims_array = np.array(total_dims)

relative_loss_test = np.array(rel_loss_test)
relative_loss_train = np.array(rel_loss_train)

relative_corr_test = np.array(rel_corr_test)
relative_corr_train = np.array(rel_corr_train)

test_rmse_SE_array = np.array(loop_test_rmse_SE)
train_rmse_SE_array = np.array(loop_train_rmse_SE)

test_corr_SE_array = np.array(loop_corr_SE_test)
train_corr_SE_array = np.array(loop_corr_SE_train)

shuffled_test_rmse_SE_array = np.array(shuffled_loop_test_rmse_SE)
shuffled_train_rmse_SE_array = np.array(shuffled_loop_train_rmse_SE)

shuffled_test_corr_SE_array = np.array(shuffled_loop_test_corr_SE)
shuffled_train_corr_SE_array = np.array(shuffled_loop_train_corr_SE)

test_rmse_EVD_array = np.array(loop_test_rmse_EVD)
train_rmse_EVD_array = np.array(loop_train_rmse_EVD)

train_corr_EVD_array = np.array(loop_corr_EVD_train)
test_corr_EVD_array = np.array(loop_corr_EVD_test)


# ---------------------------------------------------------------------
# Create performance dataframes
# ---------------------------------------------------------------------

relative_rmse_SE = (
    test_rmse_SE_array - train_rmse_SE_array
) / train_rmse_SE_array

relative_rmse_EVD = (
    test_rmse_EVD_array - train_rmse_EVD_array
) / train_rmse_EVD_array

relative_corr_SE = (
    train_corr_SE_array - test_corr_SE_array
) / train_corr_SE_array

relative_corr_EVD = (
    train_corr_EVD_array - test_corr_EVD_array
) / train_corr_EVD_array

df_rmse = pd.DataFrame(
    {
        "RMSE:test SE": test_rmse_SE_array,
        "RMSE:test EVD": test_rmse_EVD_array,
        "RMSE:train SE": train_rmse_SE_array,
        "RMSE:train EVD": train_rmse_EVD_array,
    }
)

df_corr = pd.DataFrame(
    {
        "Corr:test SE": test_corr_SE_array,
        "Corr:test EVD": test_corr_EVD_array,
        "Corr:train SE": train_corr_SE_array,
        "Corr:train EVD": train_corr_EVD_array,
    }
)

df_relative_error_se_evd = pd.DataFrame(
    {
        "Test (SE-EVD)/SE": relative_loss_test,
        "Train (SE-EVD)/SE": relative_loss_train,
    }
)

df_relative_corr_se_evd = pd.DataFrame(
    {
        "Test (EVD-SE)/SE": relative_corr_test,
        "Train (EVD-SE)/SE": relative_corr_train,
    }
)

df_performance_RMSE = pd.DataFrame(
    {
        "RMSE:test SE": test_rmse_SE_array,
        "RMSE:train SE": train_rmse_SE_array,
        "shuffled RMSE:test SE": shuffled_test_rmse_SE_array,
        "shuffled RMSE:train SE": shuffled_train_rmse_SE_array,
    }
)

df_performance_corr = pd.DataFrame(
    {
        "Corr:test SE": test_corr_SE_array,
        "Corr:train SE": train_corr_SE_array,
        "shuffled Corr:test SE": shuffled_test_corr_SE_array,
        "shuffled Corr:train SE": shuffled_train_corr_SE_array,
    }
)

df_rel = pd.DataFrame(
    {
        "RMSE:SE": relative_rmse_SE,
        "RMSE:EVD": relative_rmse_EVD,
        "Corr:SE": relative_corr_SE,
        "Corr: EVD": relative_corr_EVD,
    }
)


# ---------------------------------------------------------------------
# Save train-test reconstruction results
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    f"Data_test_train_{class_name}_BL_{nfold}fold_{runs}runs.pkl",
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

print("Saved train-test reconstruction results to:")
print(output_file)


# ---------------------------------------------------------------------
# Plot RMSE robustness
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
# Plot correlation robustness
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
