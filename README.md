The complete pipeline used to construct, validate, and analyse low-dimensional representations of MDS-UPDRS symptom data from the Parkinson's Progression Markers Initiative (PPMI) cohort.

The primary objective is to identify the intrinsic low-dimensional structure of Parkinson's disease symptoms using **Spectral Estimation (SE)** and to evaluate the stability, interpretability, and generalisability of the resulting symptom spaces. 

We supply here the code for use on data downloaded from the [PPMI repository](https://www.ppmi-info.org/), to abide by PPMI's Data Use Agreement.

We provide our discovered "baseline space" of the MDS-UPDRS as the eigenvectors in saved_data/baseline_space.pkl. 

---

# Workflow

The overall workflow is illustrated below.

```
PPMI clinical data
        │
        ▼
Patient selection
        │
        ▼
Data cleaning
        │
        ▼
Rank normalization
        │
        ▼
Spectral Estimation (SE)
        │
        ▼
Low-dimensional symptom space
        │
        ├──────────────► Reconstruction validation
        │
        ├──────────────► Robustness analysis
        │
        ├──────────────► Generalisation analysis
        │
        └──────────────► Disease progression analysis
```

---

# Folder Organisation

## 1. Data preparation

These scripts prepare the datasets used throughout the dimensionality reduction analyses.

| Script                                                           | Description                                                                                     |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `1_save_patient_data_all_2025_version.py`                        | Loads the complete PPMI dataset and saves processed patient data.                               |
| `2_normalize_and_categorize_data.py`                             | Creates the baseline dataset, removes incomplete observations, and performs rank normalization. |

---

## 2. Construction of low-dimensional spaces

These scripts estimate symptom spaces using Spectral Estimation. All spectral-estimation logic is provided by the shared `spectral_estimation.py` module (see [Spectral Estimation](#spectral-estimation)).

| Script                                       | Description                                                                                  |
| -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `3_build_low_D_space_using_SE.py`            | Constructs the baseline low-dimensional space using all baseline symptom scores.             |             |

---

## 3. Validation of the low-dimensional spaces

These scripts evaluate how well the learned spaces reconstruct unseen patient data.

| Script                                                       | Description                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------- |
| `3_leave_one_out_performance_compared_to_random.py`          | Leave-one-out reconstruction compared with shuffled-space controls. |
| `3_test_train_performance.py`                                | Repeated train-test validation of reconstruction performance.       |
| `3_plot_LOO_with_random_and_stats.py`                        | Plots leave-one-out reconstruction performance.                     |
| `3_plot_test_train_with_random_and_stats.py`                 | Plots repeated train-test reconstruction performance.               |
| `3_plot_test_train_&_leave_one_out_with_random_and_stats.py` | Combined reconstruction performance summary.                        |

---

## 4. Robustness analyses

These scripts assess the stability of the estimated low-dimensional spaces.

| Script                                     | Description                                                                |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `4_N_fold_data_subspace_creation_final.py` | Reconstructs spaces from repeated patient subsets of varying sample sizes. |
| `4_N_fold_data_subspace_plot.py`           | Plots dimensionality and subspace similarity across sample sizes.          |

---

## 5. Generalisation analyses

These scripts evaluate whether the learned spaces generalise beyond the baseline cohort.

| Script                                                  | Description                                                       |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `5_compare_genetic_BL_spaces.py`                        | Compares sporadic and genetic cohort spaces.                      |
| `5_how_good_is_BL_space_on_genetic_cohort.py`           | Tests baseline space reconstruction on the genetic cohort.        |
| `5_how_good_is_BL_space_on_later_data_all_visits.py`    | Tests baseline space reconstruction on longitudinal visits.       |

---

## 6. Disease progression analyses

These scripts project longitudinal patient trajectories into the learned symptom spaces.

| Script                             | Description                                                           |
| ---------------------------------- | --------------------------------------------------------------------- |
| `6_save_progression.py`           | Saves longitudinal patient trajectories in the low-dimensional space. |
| `6_self_defined_space.py`         | Projects patients into selected two-dimensional symptom spaces.       |
| `6_self_defined_space_plots.py`             | Produces the plots for the self-defined two-dimensional symptom spaces. |
| `6_var_across_all_6_dims.py`      | Computes variance along all six low-dimensional axes.                 |
| `6_var_across_all_6_dims_plot.py` | Visualises variance across the six dimensions.                        |

---



## 7. Helper modules

| Module                          | Purpose                                                                                              |
| ------------------------------- | --------------------------------------------------------------------------------------------------- |
| `spectral_estimation.py`        | Shared spectral-estimation routine and the single source of its analysis constants (see below).     |
| `data_set_up.py`                | Loading and preprocessing of PPMI datasets.                                                         |
| `data_formatting_functions.py`  | Rank normalization and other data processing utilities.                                             |
| `network_analysis_functions.py` | Network analysis helper functions.                                                                 |
| `network_spectra_functions.py`  | Spectral estimation helper functions.                                                              |
| `plotting_functions.py`         | Common plotting utilities.                                                                          |
| `final_figure_config.py`        | Journal figure formatting parameters.                                                              |
| `__init__.py`                   | Shared helper functions used throughout the analysis pipeline.                                     |

---

# Spectral Estimation

The low-dimensional spaces are estimated using **Spectral Estimation (SE)** applied to symptom correlation networks.

The procedure consists of:

1. Rank-normalization of symptom scores.
2. Construction of the symptom correlation matrix.
3. Formation of a weighted symptom network.
4. Estimation of the null eigenspectrum.
5. Identification of statistically significant eigenvectors.
6. Construction of the low-dimensional symptom space using the significant eigenvectors.

Only eigenvectors whose eigenvalues exceed the null distribution are retained as meaningful dimensions.

The resulting dimensions are subsequently interpreted by inspecting their symptom loadings. Labels such as *left_vs_right*, *tremor_dominant*, and *self_vs_physician* are descriptive interpretations assigned after analysis and are **not** direct outputs of the algorithm.
Spectral estimation technique is explained in detail in the paper by [Prof. Mark Humphries et al ](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254057#sec014). In this code, we used the python implementation of spectral estimation by [Thomas J Delaney](https://github.com/thomasjdelaney/Network_Noise_Rejection_Python)


## Shared implementation (`spectral_estimation.py`)

Steps 2–6 above are implemented once, in `spectral_estimation.py`, and reused by every script that builds a space (the `3_build_*` construction scripts and the `3_test_train`, `3_leave_one_out`, `4_N_fold`, and `5_compare_genetic` validation/robustness scripts).

`build_spectral_space()` returns a `SpectralSpace` object whose fields (e.g. `exceeding_space`, `exceeding_eig_vals`, `modularity_matrix`, `mean_mins_eig`, `mean_maxs_eig`) are accessed by name. When a script's correlation network is disconnected and a feature is dropped, the function raises `FeatureDroppedError`; cross-validation scripts catch this to skip the affected fold, while single-run builders let it propagate.

**Analysis constants live in one place.** The parameters that govern the estimation are defined once at the top of `spectral_estimation.py` and used as function defaults:

| Constant             | Meaning                                             | Default     |
| -------------------- | --------------------------------------------------- | ----------- |
| `NULL_MODEL_REPEATS` | Number of null networks used for the confidence bounds | `100`     |
| `CORR_METHOD`        | Correlation method used to build the network        | `"pearson"` |
| `INTERVAL_TYPE`      | Bound type passed to the null-model comparison      | `"CI"`      |
| `KEEP_POSITIVE_ONLY` | Whether negative correlations are clipped to zero   | `True`      |

Changing a value here flows through every script automatically, so all spaces are always built with identical settings. Do **not** override these at the call site (e.g. passing a different `n_null_repeats` in one script), as this would make that script's results incomparable with the rest of the pipeline.

> The identical module is also present in `../DBS_prediction_model/`, which builds its symptom spaces the same way. If you change a constant, update both copies so the two pipelines stay consistent.

---

# Generated Outputs

The scripts generate several intermediate datasets within the `saved_data/` directory, including:

* Processed baseline datasets
* Earliest ON and OFF datasets
* Low-dimensional spaces
* Reconstruction performance results
* Robustness analyses
* Disease progression projections

Publication-quality figures are saved to the `figures/` directory.

---

# Recommended Execution Order

The recommended execution order is:

```
1_save_patient_data_all_2025_version.py

2_normalize_and_categorize_data.py

3_build_low_D_space_using_SE.py

3_leave_one_out_performance_compared_to_random.py
3_test_train_performance.py
4_N_fold_data_subspace_creation_final.py

5_compare_genetic_BL_spaces.py
5_how_good_is_BL_space_on_genetic_cohort.py
5_how_good_is_BL_space_on_later_data_all_visits.py
5_how_good_is_ON_OFF_updrs_iii_space_on_later_data.py

6_save_progression.py
6_self_defined_space.py
6_self_defined_space_plots.py
6_var_across_all_6_dims.py
6_var_across_all_6_dims_plot.py


---

# Dependencies


The analysis requires the following Python packages (import name → pip package):

* numpy
* pandas
* scipy
* scikit-learn (`sklearn`)
* matplotlib
* seaborn
* kneed — knee/elbow detection, used in `data_formatting_functions.py`
* bctpy (`bct`) — Brain Connectivity Toolbox, Python port; used in `network_analysis_functions.py` (<https://pypi.org/project/bctpy/>)
* cmocean — perceptually-uniform colourmaps, used in `final_figure_config.py`

`pickle` is part of the Python standard library and does not need to be installed.

Install everything with:

```
py -m pip install numpy pandas scipy scikit-learn matplotlib seaborn kneed bctpy cmocean
```
---

# Notes

Most scripts save intermediate outputs to the `saved_data/` directory so that downstream analyses can be performed without repeating computationally intensive preprocessing steps.

Several scripts prompt the user to select either the **ON** or **OFF** medication state before execution. Ensure that the required intermediate files have been generated before running downstream analyses.
