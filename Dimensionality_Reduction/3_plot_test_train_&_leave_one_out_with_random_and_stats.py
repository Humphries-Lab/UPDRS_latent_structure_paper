
"""
Plot combined reconstruction performance for leave-one-out and 5-fold analyses.

This script loads reconstruction results from:

    1. Leave-one-out cross-validation
    2. Repeated 5-fold cross-validation

For each analysis, reconstruction performance is compared between:

    1. The true baseline low-dimensional space
    2. A shuffled eigenvector-space control

Performance is shown separately for test/held-out data and training data using:
    - RMSE
    - correlation between original and reconstructed symptom profiles

Paired statistical comparisons are reported using both paired t-tests and
Wilcoxon signed-rank tests. The paired t-test p-value is used for plot
annotation.

Outputs
-------
figures/
    RMSE_combined_violin_with_random_and_shuffled.svg
    RMSE_combined_violin_with_random_and_shuffled.png
    Corr_combined_violin_with_random_and_shuffled.svg
    Corr_combined_violin_with_random_and_shuffled.png
    RMSE_combined_violin_with_random_and_shuffled_train.svg
    RMSE_combined_violin_with_random_and_shuffled_train.png
    Corr_combined_violin_with_random_and_shuffled_train.svg
    Corr_combined_violin_with_random_and_shuffled_train.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

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

def paired_test(arr1, arr2, label1, label2, metric_name):
    """
    Run paired statistical tests between two matched performance arrays.

    NaN pairs are removed before testing.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Matched arrays to compare.

    label1, label2 : str
        Labels for the two groups being compared.

    metric_name : str
        Name of the metric being tested.

    Returns
    -------
    p_val : float
        Paired t-test p-value.

    p_val_w : float
        Wilcoxon signed-rank test p-value.
    """

    mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    arr1_filt = arr1[mask]
    arr2_filt = arr2[mask]

    if len(arr1_filt) != len(arr2_filt):
        print(
            "Warning: After filtering, arrays have unequal lengths: "
            f"{len(arr1_filt)} vs {len(arr2_filt)}"
        )

    print(f"\n{metric_name} comparison:\n{label1} vs {label2}")
    print(f"Sample size: {len(arr1_filt)}")

    t_stat, p_val = ttest_rel(arr1_filt, arr2_filt)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3e}")

    w_stat, p_val_w = wilcoxon(arr1_filt, arr2_filt)
    print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p = {p_val_w:.3e}")

    return p_val, p_val_w


def plot_combined_violin(
    dataframe,
    y_col,
    y_label,
    filename,
    palette,
    ylabel_suffix,
    pairs_to_compare=None,
):
    """
    Plot violin plots comparing true and shuffled low-dimensional spaces.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Long-format dataframe containing the metric and group label.

    y_col : str
        Name of the metric column to plot.

    y_label : str
        Label for the y-axis.

    filename : str
        Output filename without extension.

    palette : list
        Colour palette for the violin plots.

    ylabel_suffix : str
        Suffix appended to the y-axis label.

    pairs_to_compare : list of tuple, optional
        Pairs of labels to statistically compare and annotate.
    """

    if pairs_to_compare is None:
        pairs_to_compare = []

    sns.set_theme(
        style="whitegrid",
        rc={
            "xtick.labelsize": 5,
            "ytick.labelsize": fig_properties["fontsize"],
        },
    )

    fig = plt.figure(figsize=(10 / 2.5, 6 / 2.5))

    ax = sns.violinplot(
        x="Type",
        y=y_col,
        data=dataframe,
        palette=palette,
    )

    plt.ylabel(f"{y_label}-{ylabel_suffix}", fontsize=fig_properties["fontsize"])
    plt.xlabel("")
    plt.xticks(fontsize=fig_properties["fontsize"], rotation=45)

    # Draw canvas so x tick labels are populated before editing font size.
    fig.canvas.draw()

    for label in ax.get_xticklabels():
        label.set_fontsize(10)

    plt.yticks(fontsize=fig_properties["fontsize"])

    if y_col == "Correlation":
        plt.ylim(0, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    x_labels = [
        "Leave out\nBL space",
        "Leave out\nShuffled space",
        "5 fold cross validation\nBL space",
        "5 fold cross validation\nShuffled space",
    ]

    x_pos = {
        label: index
        for index, label in enumerate(x_labels)
    }

    y_max = dataframe[y_col].max()
    y_min = dataframe[y_col].min()
    y_range = y_max - y_min
    line_offset = y_range * 0.1

    for index, (label1, label2) in enumerate(pairs_to_compare):

        arr1 = dataframe.loc[
            dataframe["Type"] == label1,
            y_col,
        ].values

        arr2 = dataframe.loc[
            dataframe["Type"] == label2,
            y_col,
        ].values

        p_val, p_val_w = paired_test(
            arr1,
            arr2,
            label1,
            label2,
            y_label,
        )

        # Use paired t-test p-value for plot annotation.
        p = p_val

        x1 = x_pos[label1]
        x2 = x_pos[label2]

        y = y_max + line_offset * (index + 1)
        y = 1 if y > 1 else y

        ax.plot(
            [
                x1,
                x1,
                x2,
                x2,
            ],
            [
                y - line_offset * 0.05,
                y,
                y,
                y - line_offset * 0.05,
            ],
            lw=1.5,
            c="k",
        )

        if p < 0.001:
            text = "p < 0.001"
        else:
            text = f"p = {p:.3f}"

        ax.text(
            (x1 + x2) * 0.5,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=fig_properties["fontsize"] - 2,
        )

    plt.savefig(
        os.path.join(figures_dir, filename + ".svg"),
        format="svg",
        dpi=fig_properties["dpi"],
    )

    plt.savefig(
        os.path.join(figures_dir, filename + ".png"),
        format="png",
        dpi=fig_properties["dpi"],
    )

    plt.show(block=False)


# ---------------------------------------------------------------------
# Load leave-one-out and 5-fold cross-validation results
# ---------------------------------------------------------------------

leave_one_file = os.path.join(
    saved_data_dir,
    "Data_Leave_one_out_sporadic_BL_and_random_space.pkl",
)

with open(leave_one_file, "rb") as file:
    data_leave_one = pd.read_pickle(file)

test_train_file = os.path.join(
    saved_data_dir,
    "Data_test_train_sporadic_BL_5fold_50runs.pkl",
)

with open(test_train_file, "rb") as file:
    data_test_train = pd.read_pickle(file)


# ---------------------------------------------------------------------
# Prepare held-out/test data performance
# ---------------------------------------------------------------------

rmse_sources = [
    (data_leave_one["df_rmse"]["RMSE"].values, "Leave out\nBL space"),
    (
        data_leave_one["df_rmse"]["RMSE:random test SE"].values,
        "Leave out\nShuffled space",
    ),
    (
        data_test_train["df_performance_RMSE"]["RMSE:test SE"].values,
        "5 fold cross validation\nBL space",
    ),
    (
        data_test_train["df_performance_RMSE"]["shuffled RMSE:test SE"].values,
        "5 fold cross validation\nShuffled space",
    ),
]

corr_sources = [
    (data_leave_one["df_corr"]["Correlation"].values, "Leave out\nBL space"),
    (
        data_leave_one["df_corr"]["Corr:random test SE"].values,
        "Leave out\nShuffled space",
    ),
    (
        data_test_train["df_performance_corr"]["Corr:test SE"].values,
        "5 fold cross validation\nBL space",
    ),
    (
        data_test_train["df_performance_corr"]["shuffled Corr:test SE"].values,
        "5 fold cross validation\nShuffled space",
    ),
]

rmse_combined = pd.DataFrame(
    {
        "RMSE": np.concatenate([data for data, _ in rmse_sources]),
        "Type": sum([[label] * len(data) for data, label in rmse_sources], []),
    }
)

corr_combined = pd.DataFrame(
    {
        "Correlation": np.concatenate([data for data, _ in corr_sources]),
        "Type": sum([[label] * len(data) for data, label in corr_sources], []),
    }
)

pairs_to_compare = [
    ("Leave out\nBL space", "Leave out\nShuffled space"),
    (
        "5 fold cross validation\nBL space",
        "5 fold cross validation\nShuffled space",
    ),
]

palette = [
    "red",
    "blue",
    "green",
    "gray",
]

plot_combined_violin(
    rmse_combined,
    "RMSE",
    "RMSE",
    "RMSE_combined_violin_with_random_and_shuffled",
    palette,
    "Test",
    pairs_to_compare,
)

plot_combined_violin(
    corr_combined,
    "Correlation",
    "Correlation",
    "Corr_combined_violin_with_random_and_shuffled",
    palette,
    "Test",
    pairs_to_compare,
)


# ---------------------------------------------------------------------
# Prepare training data performance
# ---------------------------------------------------------------------

rmse_sources = [
    (
        data_leave_one["df_rmse"]["RMSE:train SE"].values,
        "Leave out\nBL space",
    ),
    (
        data_leave_one["df_rmse"]["RMSE:random train SE"].values,
        "Leave out\nShuffled space",
    ),
    (
        data_test_train["df_performance_RMSE"]["RMSE:train SE"].values,
        "5 fold cross validation\nBL space",
    ),
    (
        data_test_train["df_performance_RMSE"]["shuffled RMSE:train SE"].values,
        "5 fold cross validation\nShuffled space",
    ),
]

corr_sources = [
    (
        data_leave_one["df_corr"]["Corr:train SE"].values,
        "Leave out\nBL space",
    ),
    (
        data_leave_one["df_corr"]["Corr:random train SE"].values,
        "Leave out\nShuffled space",
    ),
    (
        data_test_train["df_performance_corr"]["Corr:train SE"].values,
        "5 fold cross validation\nBL space",
    ),
    (
        data_test_train["df_performance_corr"]["shuffled Corr:train SE"].values,
        "5 fold cross validation\nShuffled space",
    ),
]

rmse_combined = pd.DataFrame(
    {
        "RMSE": np.concatenate([data for data, _ in rmse_sources]),
        "Type": sum([[label] * len(data) for data, label in rmse_sources], []),
    }
)

corr_combined = pd.DataFrame(
    {
        "Correlation": np.concatenate([data for data, _ in corr_sources]),
        "Type": sum([[label] * len(data) for data, label in corr_sources], []),
    }
)

plot_combined_violin(
    rmse_combined,
    "RMSE",
    "RMSE",
    "RMSE_combined_violin_with_random_and_shuffled_train",
    palette,
    "Train",
    pairs_to_compare,
)

plot_combined_violin(
    corr_combined,
    "Correlation",
    "Correlation",
    "Corr_combined_violin_with_random_and_shuffled_train",
    palette,
    "Train",
    pairs_to_compare,
)


plt.show()
