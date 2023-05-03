# -*- coding: utf-8 -*-
"""
@author: Jan Koune
"""

import multiprocessing
import os
import dill as pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
# import multiprocess
from functools import partial
from pathlib import Path
from copy import deepcopy
from scipy import stats
from tqdm import tqdm

import numpy as np
import pandas as pd
from dynesty import utils as dyfunc
from sklearn.metrics import r2_score
from tabulate import tabulate
from taralli.parameter_estimation.base import DynestyParameterEstimator
from taralli.parameter_estimation.weighted_sample_statistics import (
    weighted_sample_covariance, weighted_sample_mean,
    weighted_sample_median_absolute_deviation, weighted_sample_quantile)
from taralli.utilities.kde import kde_1d, kde_2d

from modules import paths

#%% =========================================================================
# Model save path
# ===========================================================================
meas_path = os.path.join(paths["measurements"], "measurements.csv")

#%% ==================================================================
# UTILITY FUNCTIONS
# ====================================================================


def summary(model, param=None, tablefmt="latex_booktabs"):
    """
    --------------------------------------------------------------------------
    Note: This is the 'summary' method of prob taralli parameter estimation objects.
    With minimal modifications this can be used to save the output of parameter
    estimation in LaTeX format.
    
    By modifying the plotting functions this could also probably be used to 
    include the tabulated inference results directly in the corner plots.
    
    For help with the parameters see the documentation of the tabulate package.
    --------------------------------------------------------------------------
    
    Compute and print a summary of the posterior distribution containing
        `mean`, `median`, `standard deviation`, `25th percentile`,
        `75th percentile`, and `95th percentile`.

    In `model.output_summary` a short version of the summary is stored containing
    `mean`, `median`, and `covariance`.

    """

    head_str = "\\begin{table}[H] \n\\centering \n\\caption{} \n\\label{} \n"
    end_str = "\n\\end{table}"

    mean = weighted_sample_mean(model.posterior_sample, model.posterior_sample_weights)

    quantiles = weighted_sample_quantile(
        model.posterior_sample,
        np.array([0.25, 0.50, 0.75, 0.95]),
        model.posterior_sample_weights,
    )

    keys = ["mean", "median", "covariance"]
    summary = dict.fromkeys(keys)

    summary["median"] = quantiles[1, :]
    summary["covariance"] = weighted_sample_covariance(
        model.posterior_sample, model.posterior_sample_weights
    )
    summary["mean"] = mean

    col_names = ["", "mean", "median", "sd", "25%", "75%", "95%"]

    if param is None:
        param = np.array([f"  θ_{ii + 1}" for ii in range(mean.shape[0])]).reshape(
            -1, 1
        )
    else:
        param = np.reshape(param, (-1, 1))

    tab = np.hstack(
        (
            param,
            summary["mean"].reshape(-1, 1),
            quantiles[1, :].reshape(-1, 1),
            np.sqrt(np.diag(summary["covariance"])).reshape(-1, 1),
            quantiles[0, :].reshape(-1, 1),
            quantiles[2, :].reshape(-1, 1),
            quantiles[3, :].reshape(-1, 1),
        )
    )

    tab_str = tabulate(tab, headers=col_names, floatfmt=".2f", tablefmt=tablefmt)

    return head_str + tab_str + end_str


def model_selection(models, P0=None):
    """
    Given prob taralli statistical models and optionally the corresponding prior
    model probabilities, calculates the posterior model probabilities, bayes 
    factors and also provides the interpretation according to [REFERENCE].
    
    Needed by the summary_model_selection function that produces tabulated
    latex output of the model selection results.
    """
    Nmodels = len(models)
    model_ncall = [np.sum(m.parameter_estimation_output.ncall) for m in models]

    # If no P0 is supplied models are assumed apriori equally probable
    if P0 is None:
        P0 = np.repeat(1 / Nmodels, Nmodels)

    # Calculate the evidence, model posterior probabilities
    model_logz = [m.posterior_sample_log_evidence[-1] for m in models]
    model_z = np.exp(model_logz)
    model_z = np.asarray(model_z)
    total_z = np.sum(P0 * model_z)
    model_post_prob = model_z * P0 / total_z

    # Calculate occam and data log evidence terms (Simoen et al. 2013)
    occam_logz = [-m.kl_divergence[-1] for m in models]
    data_logz = np.array(model_logz) - np.array(occam_logz)

    # Bayes factors
    bayes_factors = []
    max_prob_idx = np.argmax(model_post_prob)
    for i, p_m in enumerate(model_post_prob):
        bayes_factors.append(
            model_post_prob[max_prob_idx]
            * P0[i]
            / (model_post_prob[i] * P0[max_prob_idx])
        )

    # Interpretation of Bayes factors
    bins = [10 ** 0, 10 ** 0.5, 10, 10 ** (3 / 2), 10 ** 2]
    interpretations = [
        "Negative",
        "Barely worth mentioning",
        "Substantial",
        "Strong",
        "Very strong",
        "Decisive",
    ]
    bin_idx = np.digitize(bayes_factors, bins)
    model_interpretations = [interpretations[idx] for idx in bin_idx]

    # Create the rows for the table by iterating over the models and appending
    # the quantities of interest
    tab = []
    for i, m in enumerate(models):
        row = []
        row.append(model_ncall[i])
        row.append(model_logz[i])
        row.append(data_logz[i])
        row.append(occam_logz[i])
        row.append(model_post_prob[i])
        row.append(bayes_factors[i])
        row.append(model_interpretations[i])

        # Stack the rows for each model
        tab.append(row)
    return tab


def summary_model_selection(
    models, P0=None, model_names=None, tablefmt="latex_booktabs"
):
    """
    --------------------------------------------------------------------------
    Note: This is the 'summary' method of prob taralli parameter estimation objects.
    With minimal modifications this can be used to save the output of parameter
    estimation in LaTeX format.
    
    This version of the function compares the results of different models.
    
    NOTES:
        A single line with the column names is passed e.g. ['Cv','Sigma','Kr' .. etc]
        For each model the summary of the posterior is obtained. Each parameter
        mean is assigned to each column sequentially. This means that if a given 
        model does not contain one of the parameters, subsequent entries will be shifted.
        This must be corrected by hand in the final table.
    --------------------------------------------------------------------------
    
    INPUT:
        models: List of statistical models from prob taralli output
        P0: List of prior model probabilities
    """

    # Initialization
    Nmodels = len(models)
    head_str = "\\begin{table}[H] \n\
        \\centering \n\
        \\caption{Log-evidence, posterior probability and Bayes factors per model.} \n\
        \\label{tab:autogenerated_model_selection} \n"
    end_str = "\n\\end{table}"
    headers = [
        "Model",
        "NFE",
        "log($\mathcal{Z}$)",
        "log($\mathcal{Z}_{data}$)",
        "log($\mathcal{Z}_{occam}$)",
        "p($\mathcal{M}$)",
        "$K$",
        "Interpretation",
    ]
    floatfmt = ("s", ".0f", ".2f", ".2f", ".2f", ".2f", ".2E", "s")

    # If no model names are supplied, assign numbers instead
    if model_names is None:
        model_names = []
        for i, m in enumerate(models):
            model_names.append("Model #" + str(i))
    model_names = np.reshape(model_names, (-1, 1))

    tab = model_selection(models, P0)

    # Stack horizontaly, reverse sort by evidence and call tabulate
    # tab = np.hstack((model_names, tab)).tolist()

    for idx, model_name in enumerate(model_names):
        tab[idx].insert(0, model_name[0])

    tab_sorted = sorted(tab, key=lambda x: float(x[2]), reverse=True)
    tab_str = tabulate(tab_sorted, headers=headers, floatfmt=floatfmt, tablefmt=tablefmt)

    return head_str + tab_str + end_str,


def dynesty_resample(stat_model, nsamples):
    # Resample from the statistical model.
    # NOTES:
    #   * The samples from inference can directly be used there is no need to
    #   resample technically. This way however the weighing of the samples is
    #   taken care of by the dynesty utility function
    #
    #   * If prob_taralli stat model, the posterior_sample_weights and
    #   posterior_sample vars are already in the required form (i.e. weights
    #   are normalized)
    #
    #   EXAMPLE (From Dynesty documentation):
    #   samples, weights = res2.samples, np.exp(res2.logwt - res2.logz[-1])
    #   new_samples = dyfunc.resample_equal(samples, weights)

    try:
        weights = stat_model.posterior_sample_weights
        samples = stat_model.posterior_sample
    except:
        weights = np.exp(stat_model.logwt - stat_model.logz[-1])
        samples = stat_model.samples

    resample = dyfunc.resample_equal(samples, weights)
    rnd_samples = np.random.randint(low=0, high=np.shape(resample)[0], size=nsamples)

    return resample[rnd_samples, :]


def get_parameter_estimates(stat_model):
    """
    Can work with either taralli or dynesty results objects
    """
    try:
        # This will work in case the data is prob_taralli results
        weights = stat_model.posterior_sample_weights
        samples = stat_model.posterior_sample
    except:
        # Otherwise assume the data is dynesty results
        samples = stat_model.samples
        weights = np.exp(stat_model.logwt - stat_model.logz[-1])

    # Get mean and median estimates
    param_mean = weighted_sample_mean(samples, weights)
    param_median = weighted_sample_quantile(samples, 0.5, weights)

    # Get MAP estimates
    n_params = np.shape(samples)[1]
    param_MAP = np.zeros(n_params)
    for i in range(n_params):
        _, density, x_mesh = kde_1d(sample_vec=samples[:, i], weight_vec=weights)
        idx_MAP = np.argmax(density)
        param_MAP[i] = x_mesh[idx_MAP]

    return param_mean, param_median[0], param_MAP


def sample_post_pred(func_post, resample_post):
    samples_post = []
    for jj, sample in tqdm(enumerate(resample_post)):
        samples_post.append(np.atleast_2d(func_post(sample)))
    return np.array(samples_post)


def func_sample_noise(th, params_model, model_key, std_meas, nsamples=1000):

    # Assign parameters to keys
    theta = get_theta_from_input(th, params_model, model_key)

    # Assemble noise and correlation matrix
    if "std_meas" in theta.keys():
        std_noise = theta["std_meas"]
        return stats.norm.rvs(
            loc=0.0,
            scale=std_noise,
        )
    else:
        std_noise = std_meas
        return stats.norm.rvs(
            loc=0.0,
            scale=std_noise,
            size=nsamples,
        )


def get_theta_from_input(th, params_model, model_key):

    if th.ndim == 1:
        theta = {
            param_key: th[param_idx] for param_idx, param_key in enumerate(params_model[model_key])
        }

    elif th.ndim == 2:
        theta = {
            param_key: th[:, param_idx] for param_idx, param_key in enumerate(params_model[model_key])
        }

    else:
        raise ValueError(f"Dimension of theta should be 1 or 2 but is {th.ndim}")

    return theta


def find_nearest(array, values):
    array = np.asarray(array)
    values = np.reshape(values, (-1, 1))
    idx = []
    for val in values:
        idx.append(np.abs(array - val).argmin())
    return idx


#%% =========================================================================
# LOADING AND SAVING
#
# Different files for saving and loading. Seperated into functions for models
# (prob_taralli statistical model objects) and data (results object of Dynesty)
# ===========================================================================


def save_post_pred(path, fname, data):
    fullfile = path / fname
    with open(fullfile, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def save_model(path, filename, model, timestamp=True):
    if timestamp:
        date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
    else:
        date_time = ""

    # Delete the loglikelihood and prior functions which can not be pickled. Pickling works fine
    # on some machines with different versions of windows 10 so this might be an OS issue.
    model_save = deepcopy(model)
    del(model_save.log_likelihood)
    del(model_save.prior_transform)

    fname = date_time + filename
    fullfile = path / fname
    with open(fullfile, "wb") as handle:
        pickle.dump(model_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load_pickle(path, filename):
    fullfile = path / filename
    with open(fullfile, "rb") as handle:
        data = pickle.load(handle)
    print(f"Loaded {filename}.")
    return data