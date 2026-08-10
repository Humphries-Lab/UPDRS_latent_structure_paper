
"""
Save disease-progression data in both original and low-dimensional spaces.

This script loads a selected low-dimensional symptom space, identifies patients
with sufficient ON and OFF follow-up visits, rank-normalizes their longitudinal
data, projects each visit into the selected low-dimensional space, and saves
progression measures.

For each selected patient, progression is quantified as:

    1. Distance from baseline in the low-dimensional space
    2. Distance from origin in the low-dimensional space
    3. Distance from baseline in the original feature space
    4. Distance from origin in the original feature space

Outputs
-------
saved_data/
    data_disease_progression_on_<ON/OFF/BL>_space_final.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd

import data_formatting_functions as data_fun


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")


# ---------------------------------------------------------------------
# User input: select low-dimensional space
# ---------------------------------------------------------------------

print("Which medication state should be used to build the progression space?")
print("Please select from the options below:")
print("1: ON data - earliest data of the patient ON PD medication")
print("2: OFF data - earliest data of the patient OFF PD medication")
print("3: BL data")

choice = input("Enter the number corresponding to your choice: ")

space_map = {
    "1": "ON",
    "2": "OFF",
    "3": "BL",
}

method = space_map.get(choice)

if method is None:
    raise ValueError("Invalid selection. Please restart and choose 1, 2, or 3.")

print(
    "Selected medication state used to build the progression space: "
    f"{method}"
)


# ---------------------------------------------------------------------
# Load selected low-dimensional space
# ---------------------------------------------------------------------

if method in ["ON", "OFF"]:
    load_file = f"data_Low_D_space_full_data_sporadic_earliest_{method}.pkl"
elif method == "BL":
    load_file = "data_Low_D_space_full_data_BL_sporadic.pkl"

with open(os.path.join(saved_data_dir, load_file), "rb") as file:
    selected_space_data = pickle.load(file)

exceeding_eig_space = selected_space_data["exceeding_eig_space"]


# ---------------------------------------------------------------------
# Load baseline low-dimensional-space data
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_sporadic.pkl"),
    "rb",
) as file:
    baseline_space_data = pickle.load(file)

df_full_data_non_normalized_BL = baseline_space_data[
    "df_full_data_non_normalized_BL_NaN_Rows_deleted"
]

dataframes_rank_norm_BL = baseline_space_data["dataframes_rank_norm"]


# ---------------------------------------------------------------------
# Define patient list for progression analysis
# ---------------------------------------------------------------------

if method in ["ON", "OFF"]:
    patients_list = set(selected_space_data["patients_list"]).intersection(
        set(baseline_space_data["BL_patients_list"])
    )
elif method == "BL":
    patients_list = set(baseline_space_data["BL_patients_list"])

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
# Select patients with at least seven ON and seven OFF visits
# ---------------------------------------------------------------------

filtered_df_off_full = df[
    (df["PATNO"].isin(patients_list))
    & (df["PDSTATE"] == "OFF")
]

filtered_df_on_full = df[
    (df["PATNO"].isin(patients_list))
    & (df["PDSTATE"] == "ON")
]

patno_counts_ALL_OFF = filtered_df_off_full["PATNO"].value_counts()
patno_counts_ALL_ON = filtered_df_on_full["PATNO"].value_counts()

patnos_with_n_visits_ALL_OFF = patno_counts_ALL_OFF[
    patno_counts_ALL_OFF >= 7
].index

patnos_with_n_visits_ALL_ON = patno_counts_ALL_ON[
    patno_counts_ALL_ON >= 7
].index

common_patnos_with_n_visits = set(patnos_with_n_visits_ALL_OFF).intersection(
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
# Rank normalize ON and OFF longitudinal data
# ---------------------------------------------------------------------

normalized_df_off = data_fun.rank_normalization_bsk(dataframes_off)
normalized_df_on = data_fun.rank_normalization_bsk(dataframes_on)

# Add metadata columns back after normalization so visits can be grouped
# and ordered by patient, medication state, and visit date.
normalized_df_on = normalized_df_on.copy()
normalized_df_on[specified_columns] = filtered_df_on[specified_columns].values

normalized_df_off = normalized_df_off.copy()
normalized_df_off[specified_columns] = filtered_df_off[specified_columns].values

# Add patient and event identifiers back to the normalized baseline data.
df_normalized_BL[specified_BL_columns] = df_full_data_non_normalized_BL[
    specified_BL_columns
]


# ---------------------------------------------------------------------
# Compute progression measures for each patient
# ---------------------------------------------------------------------

selected_patnos = common_patnos_with_n_visits

all_dimensions_off = [
    [
        []
        for _ in range(len(selected_patnos))
    ]
    for _ in range(exceeding_eig_space.shape[1])
]

all_dimensions_on = [
    [
        []
        for _ in range(len(selected_patnos))
    ]
    for _ in range(exceeding_eig_space.shape[1])
]

count = 0

projected_data_by_patno = {}

patno_to_index = {}
index_to_patno = {}

distance_from_BL = {}
distance_from_origin = {}

distance_from_BL_60D = {}
distance_from_origin_60D = {}

for patno in selected_patnos:

    count += 1
    index = count - 1

    patno_to_index[index] = patno
    index_to_patno[patno] = index

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
    dist_from_origin_BL_60D = np.linalg.norm(BL_data_for_projection)

    distance_from_BL_off = [dist_from_origin_BL]
    distance_from_BL_on = [dist_from_origin_BL]

    distance_from_origin_off = [dist_from_origin_BL]
    distance_from_origin_on = [dist_from_origin_BL]

    distance_from_BL_off_60D = [dist_from_origin_BL_60D]
    distance_from_BL_on_60D = [dist_from_origin_BL_60D]

    distance_from_origin_off_60D = [dist_from_origin_BL_60D]
    distance_from_origin_on_60D = [dist_from_origin_BL_60D]

    position_on = []
    position_off = []

    projected_visits_off = []
    projected_visits_on = []

    for _, row in patno_data_off.iterrows():

        data_for_projection = (
            row
            .drop(index=specified_columns)
            .values
            .reshape(1, -1)
        )

        _, _, projected_data = data_fun.project_and_recover(
            data_for_projection,
            exceeding_eig_space,
        )

        distance_from_BL_off.append(
            np.linalg.norm(projected_data - projected_data_BL)
        )

        distance_from_origin_off.append(
            np.linalg.norm(projected_data)
        )

        distance_from_BL_off_60D.append(
            np.linalg.norm(data_for_projection - BL_data_for_projection)
        )

        distance_from_origin_off_60D.append(
            np.linalg.norm(data_for_projection)
        )

        position_off.append(projected_data)
        projected_visits_off.append(projected_data.flatten())

        for dim in range(exceeding_eig_space.shape[1]):
            all_dimensions_off[dim][count - 1].append(
                projected_data[0, dim]
            )

    for _, row in patno_data_on.iterrows():

        data_for_projection = (
            row
            .drop(index=specified_columns)
            .values
            .reshape(1, -1)
        )

        _, _, projected_data = data_fun.project_and_recover(
            data_for_projection,
            exceeding_eig_space,
        )

        distance_from_BL_on.append(
            np.linalg.norm(projected_data - projected_data_BL)
        )

        distance_from_origin_on.append(
            np.linalg.norm(projected_data)
        )

        distance_from_BL_on_60D.append(
            np.linalg.norm(data_for_projection - BL_data_for_projection)
        )

        distance_from_origin_on_60D.append(
            np.linalg.norm(data_for_projection)
        )

        position_on.append(projected_data)
        projected_visits_on.append(projected_data.flatten())

        for dim in range(exceeding_eig_space.shape[1]):
            all_dimensions_on[dim][count - 1].append(
                projected_data[0, dim]
            )

    projected_data_by_patno[patno] = {
        "on": np.array(projected_visits_on).T,
        "off": np.array(projected_visits_off).T,
    }

    distance_from_BL[patno] = {
        "off": np.array(distance_from_BL_off),
        "on": np.array(distance_from_BL_on),
    }

    distance_from_origin[patno] = {
        "off": np.array(distance_from_origin_off),
        "on": np.array(distance_from_origin_on),
    }

    distance_from_BL_60D[patno] = {
        "off": np.array(distance_from_BL_off_60D),
        "on": np.array(distance_from_BL_on_60D),
    }

    distance_from_origin_60D[patno] = {
        "off": np.array(distance_from_origin_off_60D),
        "on": np.array(distance_from_origin_on_60D),
    }


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def prepare_data(distances, state):
    """
    Convert patient-level distance arrays to a long-format dataframe.

    Parameters
    ----------
    distances : dict
        Dictionary containing distance arrays for each patient.

    state : str
        Medication state to extract, either "off" or "on".

    Returns
    -------
    pd.DataFrame
        Dataframe with columns PATNO, Position, Value, and State.
    """

    data = []

    for patno, patient_distances in distances.items():
        for index, value in enumerate(patient_distances[state]):
            data.append(
                {
                    "PATNO": patno,
                    "Position": index + 1,
                    "Value": value,
                    "State": state,
                }
            )

    return pd.DataFrame(data)


def calculate_summary_stats(data_off=None, data_on=None):
    """
    Calculate mean and standard error for OFF and ON progression data.

    Parameters
    ----------
    data_off : pd.DataFrame, optional
        Long-format OFF-state progression dataframe.

    data_on : pd.DataFrame, optional
        Long-format ON-state progression dataframe.

    Returns
    -------
    dict
        Summary-statistic dataframes for available medication states.
    """

    summary_stats = {}

    if data_off is not None:
        mean_off = data_off.groupby("Position")["Value"].mean().reset_index()
        se_off = data_off.groupby("Position")["Value"].sem().reset_index()
        se_off["Value"].fillna(0, inplace=True)

        summary_off = pd.merge(mean_off, se_off, on="Position")
        summary_off.columns = [
            "Position",
            "Mean_off",
            "SE_off",
        ]

        summary_stats["off"] = summary_off

    if data_on is not None:
        mean_on = data_on.groupby("Position")["Value"].mean().reset_index()
        se_on = data_on.groupby("Position")["Value"].sem().reset_index()
        se_on["Value"].fillna(0, inplace=True)

        summary_on = pd.merge(mean_on, se_on, on="Position")
        summary_on.columns = [
            "Position",
            "Mean_on",
            "SE_on",
        ]

        summary_stats["on"] = summary_on

    return summary_stats


# ---------------------------------------------------------------------
# Convert progression distances to long-format dataframes
# ---------------------------------------------------------------------

data_off_BL = prepare_data(distance_from_BL, "off")
data_on_BL = prepare_data(distance_from_BL, "on")

data_origin_off = prepare_data(distance_from_origin, "off")
data_origin_on = prepare_data(distance_from_origin, "on")

data_off_BL_60D = prepare_data(distance_from_BL_60D, "off")
data_on_BL_60D = prepare_data(distance_from_BL_60D, "on")

data_origin_off_60D = prepare_data(distance_from_origin_60D, "off")
data_origin_on_60D = prepare_data(distance_from_origin_60D, "on")


# ---------------------------------------------------------------------
# Calculate summary statistics
# ---------------------------------------------------------------------

summary_on_origin = calculate_summary_stats(data_on=data_origin_on)
summary_off_origin = calculate_summary_stats(data_off=data_origin_off)

summary_on_origin_60D = calculate_summary_stats(data_on=data_origin_on_60D)
summary_off_origin_60D = calculate_summary_stats(data_off=data_origin_off_60D)

summary_on_BL = calculate_summary_stats(data_on=data_on_BL)
summary_off_BL = calculate_summary_stats(data_off=data_off_BL)

summary_on_BL_60D = calculate_summary_stats(data_on=data_on_BL_60D)
summary_off_BL_60D = calculate_summary_stats(data_off=data_off_BL_60D)


# ---------------------------------------------------------------------
# Save progression data
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    f"data_disease_progression_on_{method}_space_final.pkl",
)

with open(output_file, "wb") as file:
    pickle.dump(
        {
            "all_dimensions_on": all_dimensions_on,
            "all_dimensions_off": all_dimensions_off,
            "exceeding_eig_space": exceeding_eig_space,
            "projected_data_by_patno": projected_data_by_patno,
            "selected_patnos": selected_patnos,
            "data_off_BL": data_off_BL,
            "data_on_BL": data_on_BL,
            "patno_to_index": patno_to_index,
            "index_to_patno": index_to_patno,
            "data_origin_off": data_origin_off,
            "data_origin_on": data_origin_on,
            "data_off_BL_60D": data_off_BL_60D,
            "data_on_BL_60D": data_on_BL_60D,
            "data_origin_off_60D": data_origin_off_60D,
            "data_origin_on_60D": data_origin_on_60D,
        },
        file,
    )

print("Saved disease-progression data to:")
print(output_file)

