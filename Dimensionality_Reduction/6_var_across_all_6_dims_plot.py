
"""
Plot variance along all axes in the low-dimensional symptom space.

This script loads patient progression data projected along all axes of the
baseline low-dimensional space. It calculates the variance of patient positions
along each axis for:

    1. Baseline
    2. Latest OFF visit
    3. Latest ON visit

The axes are reversed before plotting to match the intended display order.

Outputs
-------
figures/
    variance_grouped_all_new.svg
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from final_figure_config import fig_properties


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


# ---------------------------------------------------------------------
# Load patient progression data
# ---------------------------------------------------------------------

input_file = os.path.join(
    saved_data_dir,
    "data_patient_progression_BL_earliest_latest_along_all_axes.pkl",
)

with open(input_file, "rb") as file:
    loaded_data = pickle.load(file)

df_patient_progression_space = loaded_data["df_patient_progression_space"]
selected_space_axes = loaded_data["selected_space_axes"]


# ---------------------------------------------------------------------
# Calculate variance along each low-dimensional axis
# ---------------------------------------------------------------------

def variance_along_axes(tuple_series):
    """
    Calculate variance along each axis for a series of patient coordinates.

    Parameters
    ----------
    tuple_series : pd.Series
        Series containing projected patient coordinates.

    Returns
    -------
    np.ndarray
        Variance along each low-dimensional axis.
    """

    arr = np.stack(tuple_series.values)
    variance = np.var(arr, axis=0, ddof=1)

    return variance


variance_BL_space1 = variance_along_axes(
    df_patient_progression_space["BL"]
)
print("Variance for BL in space 1:", variance_BL_space1)

variance_L_off_space1 = variance_along_axes(
    df_patient_progression_space["LATEST_OFF"]
)
print("Variance for latest OFF in space 1:", variance_L_off_space1)

variance_L_on_space1 = variance_along_axes(
    df_patient_progression_space["LATEST_ON"]
)
print("Variance for latest ON in space 1:", variance_L_on_space1)


# Reverse axis order for plotting.
variance_BL_space_correct_order = variance_BL_space1[::-1]
variance_L_off_space_correct_order = variance_L_off_space1[::-1]
variance_L_on_space_correct_order = variance_L_on_space1[::-1]
selected_space_axes_correct_order = selected_space_axes[::-1]


# ---------------------------------------------------------------------
# Plot grouped variance bar chart
# ---------------------------------------------------------------------

def visualize_variance_grouped_new(
    variance_BL,
    variance_L_off,
    variance_L_on,
    axis_labels=None,
):
    """
    Plot grouped variance bars for baseline, latest OFF, and latest ON states.

    Parameters
    ----------
    variance_BL : np.ndarray
        Variance along each axis at baseline.

    variance_L_off : np.ndarray
        Variance along each axis at the latest OFF visit.

    variance_L_on : np.ndarray
        Variance along each axis at the latest ON visit.

    axis_labels : list, optional
        Axis labels for the low-dimensional space.
    """

    if axis_labels is None:
        axis_labels = [
            "Dim 1",
            "Dim 2",
        ]

    set2_colors = [
        np.array([141, 211, 199]) / 255,
        np.array([255, 255, 179]) / 255,
        np.array([190, 186, 218]) / 255,
    ]

    fig_width_cm = 14
    fig_height_cm = 13.5

    fig, ax = plt.subplots(
        figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54)
    )

    bar_width = 0.25
    x = np.arange(len(variance_BL))

    ax.bar(
        x - bar_width,
        variance_BL,
        width=bar_width,
        label="Baseline",
        color=set2_colors[0],
    )

    ax.bar(
        x,
        variance_L_off,
        width=bar_width,
        label="Latest OFF",
        color=set2_colors[1],
    )

    ax.bar(
        x + bar_width,
        variance_L_on,
        width=bar_width,
        label="Latest ON",
        color=set2_colors[2],
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        axis_labels,
        fontsize=fig_properties["fontsize"],
        rotation=45,
    )

    ax.set_ylabel(
        "Variance",
        fontsize=fig_properties["fontsize"],
    )

    ax.legend(fontsize=fig_properties["fontsize"])

    ax.tick_params(
        axis="both",
        labelsize=fig_properties["fontsize"],
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(figures_dir, "variance_grouped_all_new.svg"),
        format="svg",
        bbox_inches="tight",
        dpi=fig_properties["dpi"],
    )

    plt.show(block=False)


visualize_variance_grouped_new(
    variance_BL_space_correct_order,
    variance_L_off_space_correct_order,
    variance_L_on_space_correct_order,
    axis_labels=selected_space_axes_correct_order,
)



plt.show()
