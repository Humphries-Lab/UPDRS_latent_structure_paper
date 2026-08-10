"""
Spectral estimation of the symptom subspace via Network Noise Rejection.

All analysis constants are defined ONCE here and used as function defaults,
so changing a value (e.g. the null-model repeats) flows through every script.
Do not pass these values at the call site unless you deliberately want a
different setting for that one run.
"""

import numpy as np
import __init__ as nnr   # re-exports both network_analysis_functions and network_spectra_functions
import data_formatting_functions as data_fun

# ── Analysis constants — change here, flows everywhere ───────────────
NULL_MODEL_REPEATS = 100        # null networks used for the confidence bounds
CORR_METHOD        = "pearson"  # correlation used to build the network
INTERVAL_TYPE      = "CI"       # bound type passed to getLowDimSpace / nodeRejection
KEEP_POSITIVE_ONLY = True       # clip negative correlations to 0


class FeatureDroppedError(ValueError):
    """Raised when getBiggestComponent drops a feature (disconnected network)."""


class SpectralSpace:
    """Result of build_spectral_space(). Access fields by name, e.g. space.exceeding_space."""
    def __init__(self, weighted_adjacency, modularity_matrix,
                 samples_eig_vals, samples_eig_vecs,
                 below_space, exceeding_space,
                 exceeding_upper_bound_inds, exceeding_eig_vals,
                 mean_mins_eig, mean_maxs_eig):
        self.weighted_adjacency         = weighted_adjacency
        self.modularity_matrix          = modularity_matrix
        self.samples_eig_vals           = samples_eig_vals
        self.samples_eig_vecs           = samples_eig_vecs
        self.below_space                = below_space
        self.exceeding_space            = exceeding_space
        self.exceeding_upper_bound_inds = exceeding_upper_bound_inds
        self.exceeding_eig_vals         = exceeding_eig_vals
        self.mean_mins_eig              = mean_mins_eig
        self.mean_maxs_eig              = mean_maxs_eig
        self.exceeding_dims             = exceeding_space.shape[1]
        self.below_dims                 = below_space.shape[1]


def build_spectral_space(train_set,
                         n_null_repeats=NULL_MODEL_REPEATS,
                         corr_method=CORR_METHOD,
                         interval_type=INTERVAL_TYPE,
                         keep_positive_only=KEEP_POSITIVE_ONLY):
    """
    Build the low-dimensional spectral-estimation space from a rank-normalised
    feature dataframe.

    Args:
        train_set: rank-normalised dataframe (rows = patients, cols = symptoms).

    Returns:
        SpectralSpace

    Raises:
        FeatureDroppedError: if a feature drops out of the biggest component.
        ValueError:          if the correlation matrix contains NaNs.
    """
    # ---- Construct positive feature-feature correlation network ----
    correlation_matrix = train_set.corr(method=corr_method)
    if correlation_matrix.isna().any().any():
        raise ValueError("There are NaNs in the correlation matrix.")

    corr_matrix_np = data_fun.convert_to_numpy(correlation_matrix)
    corr_matrix_np[np.diag_indices_from(corr_matrix_np)] = 0   # remove self-correlations
    if keep_positive_only:
        corr_matrix_np = corr_matrix_np.clip(min=0)            # keep positive correlations only

    weighted_adjacency_matrix = nnr.checkDirected(corr_matrix_np)
    weighted_adjacency_matrix, _, _, _ = nnr.getBiggestComponent(weighted_adjacency_matrix)

    if weighted_adjacency_matrix.shape != corr_matrix_np.shape:
        raise FeatureDroppedError(
            "Weighted adjacency matrix shape does not match correlation matrix shape."
        )

    # ---- Estimate low-dimensional space using Network Noise Rejection ----
    samples_eig_vals, optional_returns = nnr.getPoissonWeightedConfModel(
        weighted_adjacency_matrix, n_null_repeats,
        is_sparse=True, return_eig_vecs=True,
    )
    samples_eig_vecs = optional_returns["eig_vecs"]
    expected_wcm     = optional_returns["expected_wcm"]

    network_modularity_matrix = weighted_adjacency_matrix - expected_wcm

    (below_eig_space, _, [mean_mins_eig, _],
     exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, _],
     _, _, exceeding_eig_vals) = nnr.getLowDimSpace(
        network_modularity_matrix, samples_eig_vals, 0,
        int_type=interval_type, vary=0,
    )

    print("Number of exceeding dimensions =", exceeding_eig_space.shape[1])

    return SpectralSpace(
        weighted_adjacency=weighted_adjacency_matrix,
        modularity_matrix=network_modularity_matrix,
        samples_eig_vals=samples_eig_vals,
        samples_eig_vecs=samples_eig_vecs,
        below_space=below_eig_space,
        exceeding_space=exceeding_eig_space,
        exceeding_upper_bound_inds=exceeding_upper_bound_inds,
        exceeding_eig_vals=exceeding_eig_vals,
        mean_mins_eig=mean_mins_eig,
        mean_maxs_eig=mean_maxs_eig,
    )


def identify_signal_nodes(space, interval_type=INTERVAL_TYPE):
    """
    Run node rejection on a SpectralSpace to identify the signal nodes.
    Only needed by scripts that use signal-node indices; returns a dict.
    """
    reject_dict = nnr.nodeRejection(
        space.modularity_matrix, space.samples_eig_vals, 0, space.samples_eig_vecs,
        weight_type="linear", norm="L2", int_type=interval_type, bounds="upper",
    )

    signal_adjacency = space.weighted_adjacency[reject_dict["signal_inds"]][:, reject_dict["signal_inds"]]
    biggest_signal_comp, biggest_signal_inds, _, _ = nnr.getBiggestComponent(signal_adjacency)

    signal_comp_inds = reject_dict["signal_inds"][biggest_signal_inds]
    strength_distn   = biggest_signal_comp.sum(axis=0)
    leaf_inds        = np.flatnonzero(strength_distn == 1)
    keep_inds        = np.flatnonzero(strength_distn > 1)

    return {
        "reject_dict":              reject_dict,
        "signal_final_inds":        signal_comp_inds[keep_inds],
        "signal_leaf_inds":         signal_comp_inds[leaf_inds],
        "final_weighted_adjacency": biggest_signal_comp[keep_inds][:, keep_inds],
    }