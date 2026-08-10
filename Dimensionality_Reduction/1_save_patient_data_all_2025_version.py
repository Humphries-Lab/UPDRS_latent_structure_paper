"""
Save preprocessed PPMI all-visit data for a selected patient class.

This script loads all available visit data for the chosen patient group
using the 2025 preprocessing pipeline and saves the resulting dataframes,
test names, and patient identifiers as a pickle file.

The saved file is written to:
    ../saved_data/data_all_visits_<class_name>.pkl

Patient class options commonly used:
    - 'sporadic'
    - 'genetic'
    - 'both'
"""

import os
import pickle

import data_set_up as data


# ---------------------------------------------------------------------
# User-defined settings
# ---------------------------------------------------------------------

# Select the patient group to preprocess.
# Use 'sporadic', 'genetic', or 'both' depending on the analysis.
class_name = "sporadic"


# ---------------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------------

dataframes, test_names, patno_list = data.data_pre_process_all_visits_version_2025(
    patient_class=class_name
)


# ---------------------------------------------------------------------
# Define paths
# ---------------------------------------------------------------------

# Project root is assumed to be one level above the directory
# containing this script.
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

save_dir = os.path.join(base_dir, "saved_data")
os.makedirs(save_dir, exist_ok=True)

output_filename = os.path.join(
    save_dir,
    f"data_all_visits_{class_name}.pkl"
)


# ---------------------------------------------------------------------
# Save processed data
# ---------------------------------------------------------------------

processed_data = {
    "dataframes": dataframes,
    "test_names": test_names,
    "patno_list_new": patno_list,
}

with open(output_filename, "wb") as file:
    pickle.dump(processed_data, file)

print(f"Saved all-visit {class_name} data to: {output_filename}")