"""
Extract and rank-normalize baseline MDS-UPDRS data.

This script loads all-visit PPMI data for a selected patient class,
selects baseline visit data, removes treated baseline patients where
required, removes metadata columns, excludes rows with missing values,
applies rank normalization, and saves the processed baseline dataset.

Outputs are saved to:
    ../saved_data/data_baseline_<class_name>.pkl
"""

import os
import pickle

import data_formatting_functions as data_fun


# ---------------------------------------------------------------------
# Define project paths
# ---------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

saved_data_dir = os.path.join(base_dir, "saved_data")


# ---------------------------------------------------------------------
# User-defined settings
# ---------------------------------------------------------------------

# Select patient class.
# Common options are 'sporadic', 'genetic', or 'both'.
class_name = "genetic"


# ---------------------------------------------------------------------
# Load all-visit data
# ---------------------------------------------------------------------

input_file = os.path.join(
    saved_data_dir,
    f"data_all_visits_{class_name}.pkl",
)

with open(input_file, "rb") as file:
    loaded_data = pickle.load(file)

test_names = loaded_data["test_names"]
dataframes = loaded_data["dataframes"]

# Remove Modified Schwab & England ADL score because this analysis uses
# MDS-UPDRS items only.
dataframes = dataframes.drop(columns=["MSEADLG"])

test_names.remove(
    "E_Modified_Schwab___England_Activities_of_Daily_Living.csv"
)

# Keep a full non-normalized copy before baseline filtering.
df_non_normalized_full = dataframes.copy()


# ---------------------------------------------------------------------
# Select baseline visit data
# ---------------------------------------------------------------------

# Keep only baseline visit records.
dataframes_BL = df_non_normalized_full[
    df_non_normalized_full["EVENT_ID"] == "BL"
].copy()


# ---------------------------------------------------------------------
# Remove treated patients at baseline
# ---------------------------------------------------------------------

# For genetic PPMI data, some baseline records may already indicate
# PD treatment. These patients are excluded here so that the baseline
# sample represents untreated baseline assessments.
patnos_with_PDTRTMNT_1 = dataframes_BL.loc[
    dataframes_BL["PDTRTMNT"] == 1,
    "PATNO",
].unique()

dataframes_BL = dataframes_BL[
    ~dataframes_BL["PATNO"].isin(patnos_with_PDTRTMNT_1)
]


# Alternative option:
# If the analysis requires keeping only one medication-state record
# at baseline instead of excluding treated patients, replace the block
# above with a PDSTATE filter, for example:
#
#     dataframes_BL = dataframes_BL[dataframes_BL["PDSTATE"] != "OFF"]
#
# This may be relevant for genetic PPMI data where paired medication-state
# labels can occur at baseline.


# Keep a non-normalized copy before removing metadata columns.
dataframes_BL_non_normalized = dataframes_BL.copy()


# ---------------------------------------------------------------------
# Remove metadata columns before rank normalization
# ---------------------------------------------------------------------

# Columns that are not clinical score variables and should not be normalized.
metadata_columns = [
    "PDSTATE",
    "EVENT_ID",
    "INFODT",
    "DBSYN",
    "PDTRTMNT",
    "PDMEDYN",
    "PATNO",
]

# These columns are not represented in test_names and are removed before
# matching dataframe columns to test_names.
constant_columns = [
    "PATNO",
    "EVENT_ID",
    "INFODT",
]

# This reference dataframe is used only to identify which test_names
# correspond to metadata columns that are removed.
test_name_reference_df = dataframes_BL_non_normalized.copy()
test_name_reference_df.drop(columns=constant_columns, inplace=True)

indices_to_remove = []

for column in metadata_columns:
    if column in dataframes_BL_non_normalized.columns:

        # Remove metadata columns from the dataframe used for normalization.
        dataframes_BL.drop(columns=[column], inplace=True)

        # Track the corresponding test_names index, if applicable.
        if column in test_name_reference_df.columns:
            column_index = test_name_reference_df.columns.get_loc(column)
            indices_to_remove.append(column_index)

# Remove test_names entries corresponding to removed metadata columns.
test_names = [
    name
    for index, name in enumerate(test_names)
    if index not in indices_to_remove
]


# ---------------------------------------------------------------------
# Remove rows with missing values
# ---------------------------------------------------------------------

# Identify patients with missing values in any score column.
na_indices = dataframes_BL[
    dataframes_BL.isna().any(axis=1)
].index

# Remove missing rows from the score-only dataframe.
dataframes_BL.dropna(inplace=True)
dataframes_BL.reset_index(drop=True, inplace=True)

# Remove the same rows from the non-normalized dataframe so that both
# outputs contain the same patients in the same order.
dataframes_BL_non_normalized.drop(index=na_indices, inplace=True)
dataframes_BL_non_normalized.reset_index(drop=True, inplace=True)

patients_list = dataframes_BL_non_normalized["PATNO"]


# ---------------------------------------------------------------------
# Rank normalize baseline score data
# ---------------------------------------------------------------------

dataframes_BL_rank_norm = data_fun.rank_normalization_bsk(
    dataframes_BL
)


# ---------------------------------------------------------------------
# Save processed baseline data
# ---------------------------------------------------------------------

output_file = os.path.join(
    saved_data_dir,
    f"data_baseline_{class_name}.pkl",
)

with open(output_file, "wb") as file:
    pickle.dump(
        {
            "df_full_data_non_normalized_BL_NaN_Rows_deleted": dataframes_BL_non_normalized,
            "dataframes_BL_rank_norm": dataframes_BL_rank_norm,
            "test_names": test_names,
            "BL_patients_list": patients_list,
        },
        file,
    )

print("Saved baseline data to:")
print(output_file)