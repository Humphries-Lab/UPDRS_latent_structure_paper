
"""
Estimate the low-dimensional structure of baseline MDS-UPDRS data using
Network Noise Rejection (NNR).

Description
-----------
This script estimates the low-dimensional representation of
baseline rank-normalized MDS-UPDRS data using spectral estimation.
A feature-feature Pearson correlation matrix is constructed,
converted into a weighted network, and compared against a Poisson weighted
configuration null model to identify statistically significant eigenvectors.
The resulting low-dimensional space is then saved for use in downstream
analyses.

The script also:
    - Generates eigenvalue comparison figures against the null model.
    - Saves the estimated low-dimensional space and associated metadata.
    - Produces loading plots for each significant dimension, with features
      colour-coded according to clinical symptom groups.

Inputs
------
- data_baseline_<class_name>.pkl

Outputs
-------
- data_Low_D_space_full_data_BL_<class_name>.pkl
- Eigenvalue comparison figures.
- Loading plots for each estimated dimension.


"""


import data_set_up as data
import numpy as np

import data_formatting_functions as data_fun
import matplotlib.pyplot as plt
import __init__ as nnr  # network noise rejection

import pickle
import os


# Set project directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
saved_data_dir = os.path.join(base_dir, 'saved_data')
figures_dir = os.path.join(base_dir, 'figures')


# Select patient class
# Options used in the project include 'sporadic', 'genetic', or 'both'
class_name = 'sporadic'


#%% Load baseline data

with open(os.path.join(saved_data_dir, f'data_baseline_{class_name}.pkl'), 'rb') as f:
    data2 = pickle.load(f)

df_full_data_non_normalized_BL_NaN_Rows_deleted = data2['df_full_data_non_normalized_BL_NaN_Rows_deleted']
dataframes_rank_norm = data2['dataframes_BL_rank_norm']
test_names = data2['test_names']
BL_patients_list = data2['BL_patients_list']


#%% Low-dimensional space estimation

# Use rank-normalized baseline data as the input for dimensionality reduction
train_set = dataframes_rank_norm

import spectral_estimation as se

space = se.build_spectral_space(train_set)

# names used downstream in this script:
exceeding_eig_space       = space.exceeding_space
exceeding_space_dims      = space.exceeding_dims
spec_est_space            = space.exceeding_space
network_modularity_matrix = space.modularity_matrix
samples_eig_vals          = space.samples_eig_vals
mean_mins_eig             = space.mean_mins_eig
mean_maxs_eig             = space.mean_maxs_eig
exceeding_eig_vals        = space.exceeding_eig_vals


from final_figure_config import fig_properties, apply_figure_properties


#%% Plot eigenvalues against null-model eigenvalue distribution

fig1 = plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)

nnr.plotModEigValsVsNullEigHist(
    network_modularity_matrix,
    samples_eig_vals,
    mean_mins_eig,
    mean_maxs_eig
)

plt.savefig(
    os.path.join(figures_dir, f'poster_mod_eig_vals_vs_null_eig_hist_{class_name}.svg'),
    format='svg',
    dpi=fig_properties['dpi'],
    transparent=True
)

plt.show(block=False)


#%% Plot modularity eigenvalues against null-model bounds

fig2 = plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)

nnr.plotModEigValsVsNullEig(
    network_modularity_matrix,
    mean_mins_eig,
    mean_maxs_eig
)

apply_figure_properties(fig2, fig_properties)

plt.savefig(
    os.path.join(figures_dir, f'poster_mod_eig_vals_vs_null_eig_{class_name}.svg'),
    format='svg',
    dpi=fig_properties['dpi'],
    transparent=True
)

plt.show(block=False)


#%% Save baseline low-dimensional space results

with open(os.path.join(saved_data_dir, f'data_Low_D_space_full_data_BL_{class_name}.pkl'), 'wb') as f:
    pickle.dump({
        'df_full_data_non_normalized_BL_NaN_Rows_deleted': df_full_data_non_normalized_BL_NaN_Rows_deleted,
        'dataframes_rank_norm': dataframes_rank_norm,
        'exceeding_eig_space': exceeding_eig_space,
        'test_names': test_names,
        'BL_patients_list': BL_patients_list,
        'eigen_vals': exceeding_eig_vals
    }, f)


#%% Prepare feature grouping for loading plots

# Find positions where the source UPDRS/test file changes
change_positions, change_vector = data.find_boundary(test_names)

# Keep test file names in their original order, without duplicates
unique_values_in_order = []

for value in test_names:
    if value not in unique_values_in_order:
        unique_values_in_order.append(value)


# Shorter names for plotting
name_mapping = {
    'C_MDS_UPDRS_Part_II__Patient_Questionnaire.csv': 'UPDRS_II_PQ',
    'E_Modified_Schwab___England_Activities_of_Daily_Living.csv': 'Schwab & England',
    'B_MDS-UPDRS_Part_I_Patient_Questionnaire.csv': 'UPDRS_I_PQ',
    'A_MDS-UPDRS_Part_I.csv': 'UPDRS_I',
    'D_MDS-UPDRS_Part_III.csv': 'UPDRS_III'
}

unique_values_in_order2 = [name_mapping[name] for name in unique_values_in_order]


# Clinical feature groups used for colouring loading plots
Bradykinesia = [
    'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL',
    'NP3PRSPR', 'NP3PRSPL', 'NP3TTAPR', 'NP3TTAPL',
    'NP3LGAGR', 'NP3LGAGL', 'NP3BRADY'
]

Rigidity = [
    'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
    'NP3RIGN'
]

Tremor = [
    'NP3PTRMR', 'NP3KTRMR', 'NP3KTRML', 'NP3RTALJ',
    'NP3PTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL',
    'NP3RTALL', 'NP3RTCON'
]

Axial_symptoms = [
    'NP3SPCH', 'NP3POSTR', 'NP3FRZGT', 'NP3PSTBL',
    'NP3FACXP', 'NP3RISNG', 'NP3GAIT'
]

cognitive_features = [
    'NP1COG', 'NP1HALL', 'NP1DPRS', 'NP1ANXS', 'NP1APAT',
    'NP1DDS'
]

motor_features = [
    'NP2SPCH', 'NP2SALV', 'NP2WALK', 'NP2EAT', 'NP2TURN', 'NP2TRMR', 'NP2RISE',
    'NP2FREZ', 'NP2HWRT', 'NP2SWAL', 'NP1PAIN', 'NP1URIN',
    'NP1CNST', 'NP1LTHD'
]

Hoehn_Yahr = ['NHY']

ambiguous = ['NP1SLPN', 'NP1SLPD', 'NP1FATG', 'NP2DRES', 'NP2HYGN', 'NP2HOBB']


# Map each feature to a colour based on its symptom group
feature_color_map = {}

color_map = {
    'Bradykinesia': '#FFD580',
    'Rigidity': '#ADD8E6',
    'Tremor': '#98FB98',
    'Axial Symptoms': '#DDA0DD',
    'Cognitive': '#FFA07A',
    'Motor': '#40E0D0',
    'Hoehn and Yahr': '#4682B4',
    'Ambiguous': '#808080'
}


# Populate the feature-colour map
for feature in Bradykinesia:
    feature_color_map[feature] = color_map['Bradykinesia']

for feature in Rigidity:
    feature_color_map[feature] = color_map['Rigidity']

for feature in Tremor:
    feature_color_map[feature] = color_map['Tremor']

for feature in Axial_symptoms:
    feature_color_map[feature] = color_map['Axial Symptoms']

for feature in cognitive_features:
    feature_color_map[feature] = color_map['Cognitive']

for feature in motor_features:
    feature_color_map[feature] = color_map['Motor']

for feature in Hoehn_Yahr:
    feature_color_map[feature] = color_map['Hoehn and Yahr']

for feature in ambiguous:
    feature_color_map[feature] = color_map['Ambiguous']


#%% Plot loadings for each low-dimensional axis

plot_loadings = 1

if plot_loadings == 1:

    for dim in range(exceeding_space_dims):

        # Plot dimensions in reverse order for labelling
        actual_dim = exceeding_space_dims - dim

        fig, ax_main = plt.subplots(figsize=(21/2.54, 4.8/2.54), dpi=300)

        # Loadings for the current dimension
        values = spec_est_space[:, dim]

        # Colour bars by clinical feature group
        bar_colors = [
            feature_color_map.get(feature, 'gray')
            for feature in dataframes_rank_norm.columns
        ]

        bars = plt.bar(range(len(values)), values, color=bar_colors)

        # Add feature labels for stronger loadings
        for i, bar in enumerate(bars):
            if abs(values[i]) > 0.05:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    dataframes_rank_norm.columns[i],
                    ha='center',
                    va='center',
                    rotation=90,
                    fontsize=fig_properties['barlabelsize']
                )

        # Create legend handles for symptom groups
        handles = []

        for label, color in color_map.items():
            handle = plt.Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                label=label,
                markerfacecolor=color
            )
            handles.append(handle)

        plt.xlabel('Feature Index', fontsize=8)
        plt.ylabel(f'Dimension {actual_dim}', fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        # Remove x-axis labels from all except the bottom plot
        if actual_dim != exceeding_space_dims:
            plt.xlabel('')
            plt.xticks([])

        # Add secondary axis for visual separation of UPDRS/test sections
        ax_change = plt.twinx()
        ax_change.set_ylim(plt.gca().get_ylim())
        ax_change.yaxis.set_visible(False)

        # Remove unnecessary spines
        for spine in ['right', 'top']:
            ax_main.spines[spine].set_visible(False)
            ax_change.spines[spine].set_visible(False)

        # Add vertical dashed lines at test-section boundaries
        for position in change_positions:
            ax_change.axvline(x=position - 0.5, color='b', linestyle='--')

        # Add section labels above the bars for the top loading plot
        if actual_dim == 1:
            for i, (start, end) in enumerate(zip(change_positions[:-1], change_positions[1:]), start=1):

                section_title = unique_values_in_order2[i - 1]
                section_x = ((start + end) / 2) - 0.5
                section_y = 0.95

                plt.annotate(
                    section_title,
                    xy=(section_x, section_y),
                    fontsize=fig_properties['barlabelsize'],
                    ha='center',
                    va='top',
                    rotation='horizontal'
                )

        plt.tight_layout()

        # Save loading plot for the current dimension
        plt.savefig(
            os.path.join(figures_dir, f'{class_name}_dimension_{actual_dim}_plot.svg'),
            format='svg',
            dpi=fig_properties['dpi']
        )

        plt.savefig(
            os.path.join(figures_dir, f'{class_name}_dimension_{actual_dim}_plot.png'),
            format='png',
            dpi=fig_properties['dpi']
        )

    plt.show(block=False)

plt.show()
