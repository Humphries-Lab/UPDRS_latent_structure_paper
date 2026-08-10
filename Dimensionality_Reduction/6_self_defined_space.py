
"""
Save patient progression in two self-defined low-dimensional spaces.

This script loads the baseline low-dimensional symptom space and all-visit
sporadic data, identifies patients with at least seven ON and seven OFF visits,
and projects each patient's baseline, earliest ON/OFF, and latest ON/OFF visits
onto two manually selected two-dimensional spaces.

Outputs
-------
saved_data/
    data_patient_progression_selected_space_combination2.pkl
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
# Load baseline low-dimensional space
# ---------------------------------------------------------------------

with open(
    os.path.join(saved_data_dir, "data_Low_D_space_full_data_BL_sporadic.pkl"),
    "rb",
) as file:
    data_BL = pickle.load(file)

df_full_data_non_normalized_BL = data_BL[
    "df_full_data_non_normalized_BL_NaN_Rows_deleted"
]

dataframes_rank_norm_BL = data_BL["dataframes_rank_norm"]
exceeding_eig_space = data_BL["exceeding_eig_space"]

patients_list = set(data_BL["BL_patients_list"])


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
# Define self-selected two-dimensional spaces
# ---------------------------------------------------------------------

selected_patnos = common_patnos_with_n_visits

loading_nature = [
    "lower_vs_upper",
    "Brady+Rig_vs_Tremr+axial",
    "cog_vs_motor",
    "tremor_dominant",
    "self_vs_physician",
    "left_vs_right",
]

space1 = [
    3,
    5,
]

space2 = [
    2,
    4,
]

selected_space1 = exceeding_eig_space[:, space1]
selected_space2 = exceeding_eig_space[:, space2]

selected_space1_axes = [
    loading_nature[index]
    for index in space1
]

selected_space2_axes = [
    loading_nature[index]
    for index in space2
]


# ---------------------------------------------------------------------
# Project baseline, earliest, and latest visits into selected spaces
# ---------------------------------------------------------------------

df_patient_progression_space1 = pd.DataFrame(
    columns=[
        "PATNO",
        "BL",
        "EARLIEST_OFF",
        "EARLIEST_ON",
        "LATEST_OFF",
        "LATEST_ON",
    ]
)

df_patient_progression_space2 = pd.DataFrame(
    columns=[
        "PATNO",
        "BL",
        "EARLIEST_OFF",
        "EARLIEST_ON",
        "LATEST_OFF",
        "LATEST_ON",
    ]
)

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

    earliest_entry_off = patno_data_off.loc[
        patno_data_off["INFODT"].idxmin()
    ]

    latest_entry_off = patno_data_off.loc[
        patno_data_off["INFODT"].idxmax()
    ]

    earliest_entry_on = patno_data_on.loc[
        patno_data_on["INFODT"].idxmin()
    ]

    latest_entry_on = patno_data_on.loc[
        patno_data_on["INFODT"].idxmax()
    ]

    patno_BL2 = df_normalized_BL[
        (df_normalized_BL["PATNO"] == patno)
        & (df_normalized_BL["EVENT_ID"] == "BL")
    ]

    earliest_entry_on_filt = earliest_entry_on.drop(
        index=specified_columns
    ).values

    earliest_entry_off_filt = earliest_entry_off.drop(
        index=specified_columns
    ).values

    latest_entry_on_filt = latest_entry_on.drop(
        index=specified_columns
    ).values

    latest_entry_off_filt = latest_entry_off.drop(
        index=specified_columns
    ).values

    patno_BL = patno_BL2.drop(
        columns=specified_BL_columns
    ).values

    projection_space1_off_earliest = earliest_entry_off_filt @ selected_space1
    projection_space2_off_earliest = earliest_entry_off_filt @ selected_space2

    projection_space1_on_earliest = earliest_entry_on_filt @ selected_space1
    projection_space2_on_earliest = earliest_entry_on_filt @ selected_space2

    projection_space1_off_latest = latest_entry_off_filt @ selected_space1
    projection_space2_off_latest = latest_entry_off_filt @ selected_space2

    projection_space1_on_latest = latest_entry_on_filt @ selected_space1
    projection_space2_on_latest = latest_entry_on_filt @ selected_space2

    projection_space1_BL = patno_BL @ selected_space1
    projection_space2_BL = patno_BL @ selected_space2

    new_row1 = pd.DataFrame(
        [
            {
                "PATNO": patno,
                "BL": tuple(projection_space1_BL.flatten()),
                "EARLIEST_OFF": tuple(
                    projection_space1_off_earliest.flatten()
                ),
                "EARLIEST_ON": tuple(
                    projection_space1_on_earliest.flatten()
                ),
                "LATEST_OFF": tuple(
                    projection_space1_off_latest.flatten()
                ),
                "LATEST_ON": tuple(
                    projection_space1_on_latest.flatten()
                ),
            }
        ]
    )

    new_row2 = pd.DataFrame(
        [
            {
                "PATNO": patno,
                "BL": tuple(projection_space2_BL.flatten()),
                "EARLIEST_OFF": tuple(
                    projection_space2_off_earliest.flatten()
                ),
                "EARLIEST_ON": tuple(
                    projection_space2_on_earliest.flatten()
                ),
                "LATEST_OFF": tuple(
                    projection_space2_off_latest.flatten()
                ),
                "LATEST_ON": tuple(
                    projection_space2_on_latest.flatten()
                ),
            }
        ]
    )

    df_patient_progression_space1 = pd.concat(
        [
            df_patient_progression_space1,
            new_row1,
        ],
        ignore_index=True,
    )

    df_patient_progression_space2 = pd.concat(
        [
            df_patient_progression_space2,
            new_row2,
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Save selected-space progression data
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    "data_patient_progression_selected_space_combination2.pkl",
)

with open(output_file, "wb") as file:
    pickle.dump(
        {
            "df_patient_progression_space1": df_patient_progression_space1,
            "selected_space1": selected_space1,
            "selected_space1_axes": selected_space1_axes,
            "df_patient_progression_space2": df_patient_progression_space2,
            "selected_space2": selected_space2,
            "selected_space2_axes": selected_space2_axes,
        },
        file,
    )

print("Saved selected-space patient progression data to:")
print(output_file)

