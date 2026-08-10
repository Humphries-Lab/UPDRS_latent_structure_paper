
"""
Evaluate how well the baseline low-dimensional space reconstructs later visits.

This script loads the low-dimensional symptom space estimated from sporadic
baseline data and applies it to all later ON and OFF visits from the same
patients.

For each visit, reconstruction quality is measured as the correlation between
the original and reconstructed symptom profile. Correlations are then plotted
across visit index separately for OFF and ON medication states.

Outputs
-------
figures/
    how_good_is_BL_space_off_all.svg
    how_good_is_BL_space_ON_all.svg
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data_formatting_functions as data_fun
from final_figure_config import fig_properties


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")
figures_dir = os.path.join(base_dir, "figures")


# ---------------------------------------------------------------------
# Load baseline low-dimensional space
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_sporadic.pkl"),
    "rb",
) as file:
    baseline_data = pickle.load(file)

df_full_data_non_normalized_BL = baseline_data[
    "df_full_data_non_normalized_BL_NaN_Rows_deleted"
]

dataframes_rank_norm_BL = baseline_data["dataframes_rank_norm"]
exceeding_eig_space = baseline_data["exceeding_eig_space"]
patients_list = baseline_data["BL_patients_list"]

BL_data = dataframes_rank_norm_BL.copy()
BL_data_array = np.asarray(BL_data)

# Convert patient identifiers to standard Python integers for matching.
patients_list = [
    int(patient)
    for patient in patients_list
]


# ---------------------------------------------------------------------
# Load all-visit sporadic data
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_all_visits_sporadic.pkl"),
    "rb",
) as file:
    all_visit_data = pickle.load(file)

df = all_visit_data["dataframes"]


# ---------------------------------------------------------------------
# Select ON and OFF visits for baseline patients
# ---------------------------------------------------------------------

filtered_df_off_full = df[
    (df["PATNO"].isin(patients_list))
    & ((df["PDSTATE"] == "OFF") | (df["PDMEDYN"] == 0))
]

filtered_df_on_full = df[
    (df["PATNO"].isin(patients_list))
    & (df["PDSTATE"] == "ON")
]

# Count available ON and OFF records for each patient.
patno_counts_ALL = filtered_df_off_full["PATNO"].value_counts()
patno_counts_ALL_ON = filtered_df_on_full["PATNO"].value_counts()

# Keep patients with at least one OFF and one ON visit.
patnos_with_n_visits_ALL = patno_counts_ALL[
    patno_counts_ALL >= 1
].index

patnos_with_n_visits_ALL_ON = patno_counts_ALL_ON[
    patno_counts_ALL_ON >= 1
].index

common_patnos_with_n_visits = set(patnos_with_n_visits_ALL).intersection(
    patnos_with_n_visits_ALL_ON
)

filtered_df_on = filtered_df_on_full[
    filtered_df_on_full["PATNO"].isin(common_patnos_with_n_visits)
].copy()

filtered_df_off = filtered_df_off_full[
    filtered_df_off_full["PATNO"].isin(common_patnos_with_n_visits)
].copy()


# ---------------------------------------------------------------------
# Remove metadata columns before rank normalization
# ---------------------------------------------------------------------

specified_columns = [
    "PATNO",
    "PDSTATE",
    "EVENT_ID",
    "INFODT",
    "DBSYN",
    "PDTRTMNT",
    "PDMEDYN",
    "MSEADLG",
]

specified_BL_columns = [
    "PATNO",
    "EVENT_ID",
]

dataframes_off = filtered_df_off.copy()
dataframes_on = filtered_df_on.copy()

df_normalized_BL = dataframes_rank_norm_BL.copy()

for column in specified_columns:
    if column in dataframes_off.columns:
        dataframes_off.drop(columns=[column], inplace=True)
        dataframes_on.drop(columns=[column], inplace=True)


# ---------------------------------------------------------------------
# Remove visits with missing values
# ---------------------------------------------------------------------

na_indices_off = dataframes_off[
    dataframes_off.isna().any(axis=1)
].index

na_indices_on = dataframes_on[
    dataframes_on.isna().any(axis=1)
].index

dataframes_off.dropna(inplace=True)
dataframes_on.dropna(inplace=True)

filtered_df_off.drop(index=na_indices_off, inplace=True)
filtered_df_on.drop(index=na_indices_on, inplace=True)

dataframes_off.reset_index(drop=True, inplace=True)
dataframes_on.reset_index(drop=True, inplace=True)

filtered_df_off.reset_index(drop=True, inplace=True)
filtered_df_on.reset_index(drop=True, inplace=True)


# ---------------------------------------------------------------------
# Rank normalize later-visit ON and OFF score data
# ---------------------------------------------------------------------

normalized_df_off = data_fun.rank_normalization_bsk(dataframes_off)
normalized_df_on = data_fun.rank_normalization_bsk(dataframes_on)

# Add metadata columns back after normalization so visits can be grouped
# and ordered by patient, medication state, and visit date.
normalized_df_on = normalized_df_on.copy()
normalized_df_on[specified_columns] = filtered_df_on[specified_columns].values

normalized_df_off = normalized_df_off.copy()
normalized_df_off[specified_columns] = filtered_df_off[specified_columns].values

# Add baseline patient and visit identifiers back to the normalized BL data.
df_normalized_BL[specified_BL_columns] = df_full_data_non_normalized_BL[
    specified_BL_columns
]


# ---------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------

def get_corr_SE(input_array, eig_space):
    """
    Compute reconstruction correlation for one patient-visit row.

    Parameters
    ----------
    input_array : np.ndarray
        One-row clinical feature matrix.

    eig_space : np.ndarray
        Low-dimensional eigenvector space used for projection and recovery.

    Returns
    -------
    float
        Correlation between original and reconstructed feature values.
    """

    _, data_reconstructed, _ = data_fun.project_and_recover(
        input_array,
        eig_space,
    )

    input_array = np.asarray(input_array, dtype=np.float64).flatten()
    data_reconstructed = np.asarray(
        data_reconstructed,
        dtype=np.float64,
    ).flatten()

    if input_array.shape != data_reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: input {input_array.shape} "
            f"vs reconstructed {data_reconstructed.shape}"
        )

    corr = np.corrcoef(data_reconstructed, input_array)[0, 1]

    return corr


# ---------------------------------------------------------------------
# Compute reconstruction correlation across visits
# ---------------------------------------------------------------------

corr_reconstruction = {}

selected_patnos = common_patnos_with_n_visits

normalized_df_off["INFODT"] = pd.to_datetime(normalized_df_off["INFODT"])
normalized_df_on["INFODT"] = pd.to_datetime(normalized_df_on["INFODT"])

for patno in selected_patnos:

    patno_data_off = normalized_df_off[
        normalized_df_off["PATNO"] == patno
    ].copy()

    patno_data_on = normalized_df_on[
        normalized_df_on["PATNO"] == patno
    ].copy()

    patno_data_off["INFODT"] = pd.to_datetime(
        patno_data_off["INFODT"],
        format="%m/%Y",
    )

    patno_data_on["INFODT"] = pd.to_datetime(
        patno_data_on["INFODT"],
        format="%m/%Y",
    )

    # Sort visits chronologically within each medication state.
    patno_data_off = patno_data_off.sort_values(by="INFODT")
    patno_data_on = patno_data_on.sort_values(by="INFODT")

    # Baseline projection is retained from the original code.
    patno_BL = df_normalized_BL[
        (df_normalized_BL["PATNO"] == patno)
        & (df_normalized_BL["EVENT_ID"] == "BL")
    ]

    BL_data_for_projection = (
        patno_BL
        .drop(columns=specified_BL_columns, axis=1)
        .values
        .reshape(1, -1)
    )

    _, _, projected_data_BL = data_fun.project_and_recover(
        BL_data_for_projection,
        exceeding_eig_space,
    )

    dist_from_origin_BL = np.linalg.norm(projected_data_BL)

    distance_from_BL_off, distance_from_BL_on = [], []
    distance_from_origin_off = [dist_from_origin_BL]
    distance_from_origin_on = [dist_from_origin_BL]

    corr_off_SE = []
    corr_on_SE = []

    for _, row in patno_data_off.iterrows():
        data_for_projection = (
            row
            .drop(index=specified_columns)
            .values
            .reshape(1, -1)
        )

        data_for_projection = np.asarray(data_for_projection)

        corr_off_SE.append(
            get_corr_SE(data_for_projection, exceeding_eig_space)
        )

    for _, row in patno_data_on.iterrows():
        data_for_projection = (
            row
            .drop(index=specified_columns)
            .values
            .reshape(1, -1)
        )

        data_for_projection = np.asarray(data_for_projection)

        corr_on_SE.append(
            get_corr_SE(data_for_projection, exceeding_eig_space)
        )

    corr_reconstruction[patno] = {
        "off": np.array(corr_off_SE),
        "on": np.array(corr_on_SE),
    }


# ---------------------------------------------------------------------
# Convert reconstruction results to long-format dataframe
# ---------------------------------------------------------------------

all_patnos = []
all_states = []
all_visit_indices = []
all_correlation_values = []

for patno, state_data in corr_reconstruction.items():

    for index, value in enumerate(state_data["off"]):
        all_patnos.append(patno)
        all_states.append("OFF")
        all_visit_indices.append(index + 1)
        all_correlation_values.append(value)

    for index, value in enumerate(state_data["on"]):
        all_patnos.append(patno)
        all_states.append("ON")
        all_visit_indices.append(index + 1)
        all_correlation_values.append(value)

df_full = pd.DataFrame(
    {
        "PATNO": all_patnos,
        "State": all_states,
        "Visit Index": all_visit_indices,
        "Correlation": all_correlation_values,
    }
)


# ---------------------------------------------------------------------
# Compute baseline reconstruction correlation
# ---------------------------------------------------------------------

_, data_reconstructed_bl, _ = data_fun.project_and_recover(
    BL_data_array,
    exceeding_eig_space,
)

BL_corr_SE = np.array(
    [
        np.corrcoef(data_reconstructed_bl[row], BL_data_array[row])[0, 1]
        for row in range(len(BL_data_array))
    ]
)

bl_median = np.median(BL_corr_SE)


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------

def polish_plot(ax, bl_median):
    """Apply common formatting to later-visit reconstruction plots."""

    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xticklabels(
        [
            r"$V_1$",
            r"$V_5$",
            r"$V_{10}$",
            r"$V_{15}$",
            r"$V_{20}$",
        ]
    )

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    ax.set_ylim(0.0, 1.05)

    ax.axhline(
        bl_median,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="BL median",
    )

    ax.legend(fontsize=9, framealpha=0.5)

    sns.despine(ax=ax)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Plot OFF visit reconstruction quality
# ---------------------------------------------------------------------

plt.figure(figsize=(12 / 2.5, 10 / 2.5), dpi=300)

ax = sns.violinplot(
    x="Visit Index",
    y="Correlation",
    data=df_full[df_full["State"] == "OFF"],
    inner="box",
    palette="Blues",
    cut=0,
    linewidth=0.8,
    width=0.7,
    scale="width",
)

plt.title("OFF Medication", fontsize=12)
plt.xlabel("Visit Index", fontsize=11)
plt.ylabel("Correlation (Original vs. Reconstructed)", fontsize=11)

polish_plot(ax, bl_median)

plt.savefig(
    os.path.join(figures_dir, "how_good_is_BL_space_off_all.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
    transparent=True,
)

plt.show(block=False)


# ---------------------------------------------------------------------
# Plot ON visit reconstruction quality
# ---------------------------------------------------------------------

plt.figure(figsize=(12 / 2.5, 10 / 2.5), dpi=300)

ax = sns.violinplot(
    x="Visit Index",
    y="Correlation",
    data=df_full[df_full["State"] == "ON"],
    inner="box",
    palette="Oranges",
    cut=0,
    linewidth=0.8,
    width=0.7,
    scale="width",
)

plt.title("ON Medication", fontsize=12)
plt.xlabel("Visit Index", fontsize=11)
plt.ylabel("Correlation (Original vs. Reconstructed)", fontsize=11)

polish_plot(ax, bl_median)

plt.savefig(
    os.path.join(figures_dir, "how_good_is_BL_space_ON_all.svg"),
    format="svg",
    dpi=fig_properties["dpi"],
    transparent=True,
)

plt.show(block=False)



plt.show()
