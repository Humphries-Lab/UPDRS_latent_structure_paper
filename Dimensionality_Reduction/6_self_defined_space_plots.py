
"""
Analyse patient progression in self-defined two-dimensional spaces.

This script loads patient progression coordinates projected into two
self-defined 2D spaces. It visualizes patient positions at:

    1. Baseline
    2. Latest OFF visit
    3. Latest ON visit

For each self-defined space, the script also computes:
    - center of mass for each disease/medication state
    - variance along each selected axis

Outputs
-------
figures/
    patient_progression_plot_<axis1>_vs_<axis2>.svg
    variance_grouped_<axis1>_<axis2>.png
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from final_figure_config import fig_properties


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


# ---------------------------------------------------------------------
# Load self-defined space progression data
# ---------------------------------------------------------------------

input_file = os.path.join(
    saved_data_dir,
    "data_patient_progression_selected_space_combination2.pkl",
)

with open(input_file, "rb") as file:
    loaded_data = pickle.load(file)

df_patient_progression_space1 = loaded_data["df_patient_progression_space1"]
df_patient_progression_space2 = loaded_data["df_patient_progression_space2"]

selected_space1_axes = loaded_data["selected_space1_axes"]
selected_space2_axes = loaded_data["selected_space2_axes"]


# ---------------------------------------------------------------------
# Plot patient positions in the selected two-dimensional space
# ---------------------------------------------------------------------

def plot_patient_progression_new(df, xlabel="Dimension X", ylabel="Dimension Y"):
    """
    Plot patient positions at baseline, latest OFF, and latest ON visits.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing patient progression coordinates in BL,
        LATEST_OFF, and LATEST_ON columns.

    xlabel : str
        Label for the x-axis.

    ylabel : str
        Label for the y-axis.
    """

    set2_colors = np.array(
        [
            [102, 194, 165],
            [252, 141, 98],
            [141, 160, 203],
        ]
    ) / 255.0

    fig_width_cm = 10.5
    fig_height_cm = 14.85

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54),
        constrained_layout=True,
    )

    x_values = []
    y_values = []

    for _, row in df.iterrows():

        x_values.extend(
            [
                row["BL"][0],
                row["LATEST_OFF"][0],
                row["LATEST_ON"][0],
            ]
        )

        y_values.extend(
            [
                row["BL"][1],
                row["LATEST_OFF"][1],
                row["LATEST_ON"][1],
            ]
        )

        axes[0, 0].scatter(
            row["BL"][0],
            row["BL"][1],
            facecolors="none",
            edgecolors=set2_colors[0],
            marker="o",
            s=20,
        )

        axes[0, 1].scatter(
            row["LATEST_OFF"][0],
            row["LATEST_OFF"][1],
            facecolors="none",
            edgecolors=set2_colors[1],
            marker="s",
            s=20,
        )

        axes[1, 1].scatter(
            row["LATEST_ON"][0],
            row["LATEST_ON"][1],
            facecolors="none",
            edgecolors=set2_colors[2],
            marker="s",
            s=20,
        )

    x_min = min(x_values)
    x_max = max(x_values)

    y_min = min(y_values)
    y_max = max(y_values)

    margin = (y_max - y_min) * 0.05

    axes[0, 0].set_title(
        "Baseline State",
        fontsize=fig_properties["largefontsize"],
    )

    axes[0, 1].set_title(
        "Latest OFF State",
        fontsize=fig_properties["largefontsize"],
    )

    axes[1, 1].set_title(
        "Latest ON State",
        fontsize=fig_properties["largefontsize"],
    )

    fig.delaxes(axes[1, 0])

    for ax in axes.flat:

        if not ax.get_visible():
            continue

        ax.axhline(
            0,
            color="black",
            linestyle="-",
            linewidth=1,
        )

        ax.axvline(
            0,
            color="black",
            linestyle="-",
            linewidth=1,
        )

        ax.set_xlabel(
            xlabel,
            fontsize=fig_properties["largefontsize"],
        )

        ax.set_ylabel(
            ylabel,
            fontsize=fig_properties["largefontsize"],
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.tick_params(
            axis="both",
            labelsize=fig_properties["largefontsize"],
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(
        os.path.join(
            figures_dir,
            f"patient_progression_plot_{xlabel}_vs_{ylabel}.svg",
        ),
        format="svg",
        bbox_inches="tight",
        dpi=fig_properties["dpi"],
    )

    plt.show(block=False)


plot_patient_progression_new(
    df_patient_progression_space1,
    selected_space1_axes[0],
    selected_space1_axes[1],
)

plot_patient_progression_new(
    df_patient_progression_space2,
    selected_space2_axes[0],
    selected_space2_axes[1],
)


# ---------------------------------------------------------------------
# Centre-of-mass calculations
# ---------------------------------------------------------------------

def com(tuple_series, axis=0):
    """
    Compute the center of mass for a series of coordinate tuples.

    Parameters
    ----------
    tuple_series : pd.Series
        Series containing coordinate tuples or arrays.

    axis : int
        Axis along which to average.

    Returns
    -------
    np.ndarray
        Center of mass coordinates.
    """

    arr = np.stack(tuple_series.values)

    return np.sum(arr, axis=axis) / arr.shape[axis]


com_space1_BL = com(df_patient_progression_space1["BL"])
print("Center of mass for BL:", com_space1_BL)

com_space1_L_off = com(df_patient_progression_space1["LATEST_OFF"])
print("Center of mass for latest OFF:", com_space1_L_off)

com_space1_L_on = com(df_patient_progression_space1["LATEST_ON"])
print("Center of mass for latest ON:", com_space1_L_on)


com_space2_BL = com(df_patient_progression_space2["BL"])
print("Center of mass for BL:", com_space2_BL)

com_space2_L_off = com(df_patient_progression_space2["LATEST_OFF"])
print("Center of mass for latest OFF:", com_space2_L_off)

com_space2_L_on = com(df_patient_progression_space2["LATEST_ON"])
print("Center of mass for latest ON:", com_space2_L_on)


# ---------------------------------------------------------------------
# Variance calculations
# ---------------------------------------------------------------------

def variance_along_axes(tuple_series):
    """
    Calculate variance along the two selected axes.

    Parameters
    ----------
    tuple_series : pd.Series
        Series containing two-dimensional coordinate tuples or arrays.

    Returns
    -------
    np.ndarray
        Variance along each selected axis.
    """

    variance = np.zeros(2)
    arr = np.stack(tuple_series.values)

    variance[0] = np.var(arr[:, 0], ddof=1)
    variance[1] = np.var(arr[:, 1], ddof=1)

    return variance


variance_BL_space1 = variance_along_axes(df_patient_progression_space1["BL"])
print("Variance for BL in space 1:", variance_BL_space1)

variance_L_off_space1 = variance_along_axes(
    df_patient_progression_space1["LATEST_OFF"]
)
print("Variance for latest OFF in space 1:", variance_L_off_space1)

variance_L_on_space1 = variance_along_axes(
    df_patient_progression_space1["LATEST_ON"]
)
print("Variance for latest ON in space 1:", variance_L_on_space1)


variance_BL_space2 = variance_along_axes(df_patient_progression_space2["BL"])
print("Variance for BL in space 2:", variance_BL_space2)

variance_L_off_space2 = variance_along_axes(
    df_patient_progression_space2["LATEST_OFF"]
)
print("Variance for latest OFF in space 2:", variance_L_off_space2)

variance_L_on_space2 = variance_along_axes(
    df_patient_progression_space2["LATEST_ON"]
)
print("Variance for latest ON in space 2:", variance_L_on_space2)


# ---------------------------------------------------------------------
# Visualize variance grouped by disease/medication state
# ---------------------------------------------------------------------

def visualize_variance_grouped(
    variance_BL,
    variance_L_off,
    variance_L_on,
    axis_labels=None,
):
    """
    Plot variance along each selected axis for BL, latest OFF, and latest ON.

    Parameters
    ----------
    variance_BL : array-like
        Variance along selected axes for baseline state.

    variance_L_off : array-like
        Variance along selected axes for latest OFF state.

    variance_L_on : array-like
        Variance along selected axes for latest ON state.

    axis_labels : list, optional
        Labels for the two selected axes.
    """

    if axis_labels is None:
        axis_labels = [
            "Dim 1",
            "Dim 2",
        ]

    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.25
    x = np.arange(2)

    ax.bar(
        x - bar_width,
        variance_BL,
        width=bar_width,
        label="BL",
        color="blue",
    )

    ax.bar(
        x,
        variance_L_off,
        width=bar_width,
        label="Latest OFF",
        color="red",
    )

    ax.bar(
        x + bar_width,
        variance_L_on,
        width=bar_width,
        label="Latest ON",
        color="green",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        axis_labels,
        fontsize=fig_properties["largefontsize"],
    )

    ax.set_ylabel(
        "Variance",
        fontsize=fig_properties["largefontsize"],
    )

    ax.set_title(
        "Variance along each axis",
        fontsize=fig_properties["largefontsize"],
    )

    ax.legend(fontsize=fig_properties["largefontsize"])

    ax.tick_params(
        axis="both",
        labelsize=fig_properties["largefontsize"],
    )

    xlabel = axis_labels[0]
    ylabel = axis_labels[1]

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            f"variance_grouped_{xlabel}_{ylabel}.png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=fig_properties["dpi"],
    )

    plt.show(block=False)


visualize_variance_grouped(
    variance_BL_space1,
    variance_L_off_space1,
    variance_L_on_space1,
    axis_labels=selected_space1_axes,
)

visualize_variance_grouped(
    variance_BL_space2,
    variance_L_off_space2,
    variance_L_on_space2,
    axis_labels=selected_space2_axes,
)



plt.show()
