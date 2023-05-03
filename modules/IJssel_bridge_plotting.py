# -*- coding: utf-8 -*-
"""
@author: Jan Koune
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

from dynesty import plotting as dyplot
from taralli.parameter_estimation.weighted_sample_statistics import (
    weighted_sample_mean,
)

from modules.IJssel_bridge_truck_data import *
from copy import deepcopy

from taralli.parameter_estimation.utils import get_credible_region_1d
from taralli.parameter_estimation.weighted_sample_statistics import (
    weighted_sample_quantile,
)
from taralli.utilities.kde import kde_1d
from matplotlib.lines import Line2D

from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

#%% =========================================================================
# PLOTTING PARAMETERS
# ===========================================================================
#colors_vec = cm.get_cmap('tab10', 10)
colors_vec = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#print(mpl.rcParams.keys())

paramsGlobal = {
   'figure.max_open_warning': 100,
   'axes.labelsize': 15,
   'legend.fontsize': 15,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   'font.family': 'Arial',
   'axes.titlesize': 'medium',
   'axes.linewidth': 0.5,
   }

paramsPosteriorLarge = {
    'axes.labelsize': 24,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'legend.fontsize': 22,
   'figure.titlesize': 22,
    }
   #'xtick.bottom': False,
   #'xtick.top': False,
   #'xtick.major.top': False,
   #'xtick.major.bottom': False,
   #'ytick.left': False,
   #'ytick.right': False,
   #'ytick.major.left': False,
   #'ytick.major.right': False,

paramsPosteriorSmall = {
    'axes.labelsize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'legend.fontsize': 12,
   'figure.titlesize': 16,
    }

paramsPosteriorReliability = {
    'axes.labelsize': 16,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'legend.fontsize': 16,
   'figure.titlesize': 16,
    }

paramsPostPred = {
        'figure.figsize': [9.0, 7.0],
        'figure.titlesize': 18,
       'axes.labelsize': 16,
       'legend.fontsize': 14,
       'xtick.labelsize': 15,
       'ytick.labelsize': 15,
       'text.usetex': False,
       'font.family': 'Arial',
       'axes.titlesize': 'x-large',
       'axes.linewidth': 0.5,
    }

paramsPostComparison = {
    'figure.figsize': [4.5, 4.5],
    'lines.linewidth': 2.0
   
    }

paramsPostIlVar = {
    'figure.figsize': [9, 8],
    'lines.linewidth': 2.0
    }

def reset_global_default():
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(paramsGlobal)
    
def update_plot_params(params):
    mpl.rcParams.update(params)
    

#%% ==================================================================
# PLOTTING FUNCTIONS
# ====================================================================
def plot_posterior(model, labels, title, transform = None, plot_range = None, ground_truth = None):
    """
    Function for additional manipulation of prob taralli posterior plots. Can
    be used to add labels, titles, and scale parameters. 
    
    scale is a list of length Nparams that defines if a parameter should be plotted
    in linear or log scale. 0 = linear, 1 = log.
    """
    # Deepcopy the stat model object in case scaling is applied. All of the 
    # manipulations and plotting are done for the copy.
    stat_model = deepcopy(model)
    
    Nparams = np.shape(stat_model.posterior_sample)[1]
    
    # Apply style
    reset_global_default()
    #mpl.rcParams.update(paramsPosteriorSmall)
    mpl.rcParams.update(paramsPosteriorReliability)
    
    # Check if transforms are None:
    if transform is None:
        transform = [None] * Nparams
        
    # Apply transformation
    for i in range(Nparams):
        if transform[i]:
            ft = transform[i]
            stat_model.posterior_sample[:, i] = ft(stat_model.posterior_sample[:, i])
    
    # Get the plot objects
    pltobj = stat_model.plot_posterior(
        plotting_range = plot_range,
        title = title,
        dim_labels = labels,
        ground_truth = ground_truth,
        kde_1d_n_x_vec = 2 ** 14,
        kde_2d_n_row_mx = 2 ** 10
        )
    fig, axes = pltobj[0]
    ax_shape = np.shape(axes)

    # Add title
    #fig.suptitle('Model: \n' + title, y = 0.75, x = 0.75)

                
    return fig, axes


def plot_post_pred(samples_post, samples_noise, vec_x, y_true, fname, std_meas=None, plot_range = None, plot_title="", subplot_titles = None, prob_mass=0.90):
    y_true = np.atleast_2d(y_true)
    """
    Plot the posterior predictive stress influence line for the IJsselbridge FE model.
    
    `samples_post` must be an array of samples from the posterior predictive stress
    distribution in the shape `(nsamples, n_out, n_sensors, n_pts)` where:
    
    `n_samples` is the number of posterior predictive samples
    `n_out` is the number of outputs influence lines per sensor (2 for left and right lanes)
    `n_sensors` is the number of sensors for which the posterior preditive will be calculated
    `n_pts` is the number of points along the influence lines
    """

    # Initialize lists for metrics
    R2_list = []
    ME_list = []
    MSE_list = []
    MAE_list = []
    PE_list = []
    RMSE_list = []

    print("==================================================")
    print(f"Posterior predictive for case:n{plot_title}")

    #
    # For each sample from the posterior theta_i, draw a sample of the
    # posterior predictive
    # Shape (npts, nout, nsensors, nx)
    # samples_post = sample_post_pred(func_post, resample_post)
    arr_mean = np.mean(samples_post, axis=0)
    idx_peak = np.argmax(arr_mean, axis=-1)


    # Get the size of the sampling function output
    samples_post_shape = np.shape(samples_post)
    N_samples = samples_post_shape[0]
    N_out = samples_post_shape[1]
    N_sensors = samples_post_shape[2]
    N_pts = samples_post_shape[3]

    print("--------------------------------------------------")
    print(f"Using {N_samples} samples")

    # Initialize plot
    fig = []
    ax = []
    for _i in range(int(N_sensors)):
        fig_i, ax_i = plt.subplots(N_out, 1)
        fig.append(fig_i)
        ax.append(ax_i)
    ax = np.array(ax)

    for j in range(N_sensors):
        row = j
        print(f"Sensor: {j}")

        for i in range(N_out):
            print(f"Output: {i}")
            # Counter for plotting.
            vec_mean = arr_mean[i, j, :]
            col = i
            print(f"KDE for output {i+1}/{int(N_out)}, sensor {subplot_titles[row]} {j+1}/{int(N_sensors)}")

            # TODO :
            # * The sampling can be done using stats.norm.rvs
            # samples_noise = func_noise(resample_post)
            _, dens_noise, y_vec_noise = kde_1d(samples_noise)
            cr_noise, p_star_noise = get_credible_region_1d(y_vec_noise, dens_noise, prob_mass=prob_mass)
            credible_regions_noise = [np.min(cr_noise[prob_mass]), np.max(cr_noise[prob_mass])]

            # For each node - x position do kernel density estimation
            credible_regions = np.zeros((N_pts, 2))
            for k in range(N_pts):
                _, dens, y_vec = kde_1d(samples_post[:, i, j, k])
                cr, p_star = get_credible_region_1d(y_vec, dens, prob_mass=prob_mass)

                # TODO : We are assuming a unimodal posterior predictive here, add a check to make sure it actually is
                credible_regions[k, :] = [np.min(cr[prob_mass]), np.max(cr[prob_mass])]

            # Include R2 in plot
            r2 = r2_score(y_true[i, j, :], arr_mean[i, j, :])
            ME = max_error(y_true[i, j, :], arr_mean[i, j, :])
            MAE = mean_absolute_error(y_true[i, j, :], arr_mean[i, j, :])
            MSE = mean_squared_error(y_true[i, j, :], arr_mean[i, j, :])
            RMSE = mean_squared_error(y_true[i, j, :], arr_mean[i, j, :], squared=False)
            PE = np.abs(y_true[i, j, idx_peak[i, j]] - arr_mean[i, j, idx_peak[i, j]])

            R2_list.append(r2)
            ME_list.append(ME)
            MAE_list.append(MAE)
            MSE_list.append(MSE)
            PE_list.append(PE)
            RMSE_list.append(RMSE)

            # Plot posterior predictive and uncertainty
            ax[row, col].fill_between(
                vec_x,
                credible_regions[:, 0],
                y2=credible_regions[:, 1],
                color="blue",
                alpha=0.5,
                label=f"Combined {prob_mass} CI (HD)",
            )

            ax[row, col].fill_between(
                vec_x,
                vec_mean + credible_regions_noise[0],
                y2= vec_mean + credible_regions_noise[1],
                color="green",
                label=f"Additive {prob_mass} CI",
                alpha=0.5,
            )
            ax[row, col].plot(vec_x, vec_mean, color="red", label="Mean prediction")
            ax[row, col].set_ylabel("Stress [MPa]")

            # Plot measurements
            ax[row, col].plot(
                vec_x, y_true[i, j, :], color="black", linestyle="dashed", label="Measurement"
            )

            ax[row, col].text(
                0.15,
                0.1,
                r"RMSE = ${{{:.3f}}}$".format(RMSE),
                fontsize=15,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax[row, col].transAxes,
            )

            # Axis properties, labels etc..
            ax[row, 0].legend()
            if plot_range is not None:
                ax[row, col].set_ylim(np.atleast_2d(plot_range)[col, :])
            if (col == 0):
                ax[row, col].set_xticklabels([])

            str_subtitle = ""
            fig[row].suptitle(f"Posterior predictive: {plot_title}")
            ax[row, 0].set_title("Truck load: Fast lane" + str_subtitle)
            ax[row, 1].set_xlabel("Front axle position - x direction [m]")
            ax[row, 1].set_title("Truck load: Slow lane")
            fig[row].savefig(
                fname=fname + "_to_" + np.atleast_1d(subplot_titles)[row]
            )

    metrics = [R2_list, ME_list, MAE_list, MSE_list, PE_list, RMSE_list]
    return fig, ax, metrics


def plot_credible_intervals(data,
                            row_names, 
                            col_names,
                            col_labels = None,
                            ground_truth = None,
                            xlabels = None,
                            figshape = None, 
                            cr_prob_mass = 0.90, 
                            kde_1d_n: int = 2 ** 14,
                            prob_mass_for_plotting_range = 0.999,
                            plot_range = None,
                            transforms = None,
                            label_fontsize = 18,
                            legend_fontsize = 15):
    """
    Plot credible intervals for rows and columns of data.
    
    NOTE: Specifically for ploting CIs of the posterior samples or posterior
    predictives. Therefore expects data to be in a specific shape:
        
        * Posterior predictive: nrow x ncol x nsample
        * Posterior: array of ncol objects with nrow x sample items each. nrow
        must be the same for each of the ncol objects
    
    This should be fixed so that it only takes one specific shape of data but
    for now it will have to do.
    
    TODO: Documentation
    """
    
    # Get row and col number. Sample size can vary per row/col
    
    # if 3d array is provided assume its shape is nrow x ncol x nsample
    transpose_flag = 0
    if np.shape(np.shape(data))[0] == 3:
        nrows = np.shape(data)[0]
        ncols = np.shape(data)[1]
    
    # Else check if it an array of nrow objects.
    elif np.shape(np.shape(data))[0] == 1:
        transpose_flag = 1
        print(np.shape(np.shape(data)))
        nrows = np.shape(data)[0]
        
        # Check same number of columns for each row
        ncol_list = []
        for i in range(nrows):
            ncol_list.append(np.shape(data[i])[1])
        
        if len(np.unique(ncol_list)) != 1:
            raise ValueError("Inconsistent number of columns: {ncoll_list}")
        else:
            ncols = np.unique(ncol_list)[0]
    
    else:
        raise ValueError("Invalid data shape")

    # Check if column labels are provided. If not then make a empty array of size ncols
    if col_labels is None:
        col_labels = [f"Col{_i}" for _i in range(ncols)]

    # Check if transforms are None:
    if transforms is None:
        transforms = [None] * ncols
    
    # Default shape if none is provided
    if figshape is None:
        fig_dim_1 = int(np.ceil(ncols/3))
        figshape = (fig_dim_1, 3)
    n_fig_tot = np.prod(figshape)
        
    fig, ax = plt.subplots(figshape[0], figshape[1], figsize = (9, 2 * figshape[0] + 3))
    ax = np.atleast_1d(ax)
    
    # Create a dict to return the credible intervals
    cr_dict = dict.fromkeys(col_names)
    
    # For each sample
    for i in range(ncols):
        cr_bounds_list = []
        # Get ax index
        ax_idx = np.unravel_index(i, np.shape(ax))
        
        plotting_range_list = []
        for j in range(nrows):
            
            # Data
            if transpose_flag == 0:
                row_data = data[j]
            else:
                 row_data = np.transpose(data[j])   

            if transforms[i]:
                sample = transforms[i](row_data[i])
            else:
                sample = row_data[i]

            weight_vec = np.ones(len(sample))
            
            # Get plotting range
            prob_mass_out = 1 - prob_mass_for_plotting_range
            quantile_prob = np.array(
                [prob_mass_out / 2, prob_mass_for_plotting_range + prob_mass_out / 2]
            )
            
            plotting_range = weighted_sample_quantile(
                sample=sample, quantile_prob=quantile_prob, weight_vec=weight_vec, axis=0
            )
            plotting_range_list.append(plotting_range)
            
            # Kernel density estimate
            # TODO: Weighted Kde
            _, density, x_mesh = kde_1d(
                sample_vec=sample, weight_vec=weight_vec, n_x_vec=int(kde_1d_n)
            )
            
            # Credible region boundaries
            cr_boundaries, p_star_list = get_credible_region_1d(
                x_vec=x_mesh,
                density_vec=density,
                prob_mass=cr_prob_mass,
            )
            
            # Get mean
            mean = weighted_sample_mean(
                sample,
                weight_vec = weight_vec
                )
            
            # Plot credible interval
            for bounds in cr_boundaries[cr_prob_mass]:
                
                obj_cr = ax[ax_idx].plot(
                        [bounds[0], bounds[1]],
                        [row_names[j], row_names[j]],
                        color = 'lightblue',
                        alpha = 0.75,
                        linewidth = 10.0,
                        )
        
            # Append the credible region bounds to the list
            cr_bounds_list.append(cr_boundaries)
            
            # Plot mean and quantile
            obj_mean = ax[ax_idx].scatter([mean], row_names[j], marker = 'd', s = 200, color = 'dodgerblue')

        # Append the CRs to the dict
        cr_dict[col_names[i]] = cr_bounds_list
    
        # Ground truth
        if ground_truth is not None:
            ground_truth = np.atleast_1d(ground_truth)
            obj_ground_truth = ax[ax_idx].axvline(ground_truth[i], color = 'red', linestyle = 'dashed')
        
        # Fix plot ranges
        rmin, rmax = np.min(plotting_range_list), np.max(plotting_range_list)

        if plot_range is None:
            plt_ext = ( rmin - (rmax - rmin)*0.2, rmax + (rmax - rmin)*0.2 )
        else:
            plt_ext = plot_range[i]
        
        # Format ax
        ax[ax_idx].set_xlim(plt_ext[0], plt_ext[1])
        ax[ax_idx].set_ylim([-0.3, nrows -1 + 0.3])
        ax[ax_idx].set_title(np.atleast_1d(col_labels)[i], fontsize=20)
        ax[ax_idx].grid(axis = 'x', which = 'both')
        ax[ax_idx].yaxis.set_visible(False) if ax_idx[-1] != 0 else ax[ax_idx].yaxis.set_visible(True)
        ax[ax_idx].tick_params(axis='both', which='major', labelsize=18)
        ax[ax_idx].xaxis.get_offset_text().set_fontsize(15)

        if xlabels is not None:
            ax[ax_idx].set_xlabel(xlabels[i], fontsize = 18)
        fig.tight_layout()
        
        # Hide empty plots
    for i in range(n_fig_tot):
        ax_idx = np.unravel_index(i, np.shape(ax))
        if i+1 > ncols:
            ax[ax_idx].axis('off')
       
        
    obj_cr = Line2D([0],
                      [0],
                      color='lightblue',
                      linewidth = 10.0,
                      )
    
    # Legend
    if ground_truth is not None:
        lgd_items = [obj_cr, obj_mean, obj_ground_truth]
    else:
        lgd_items = [obj_cr, obj_mean]
    lgd = fig.legend(lgd_items, 
               [f'{cr_prob_mass * 100}% HD credible interval', 'Mean', 'Measurements'],
               fontsize = 15, 
               bbox_to_anchor=(0.5, -0.15),
               bbox_transform=fig.transFigure,
               loc = 'lower center',
               )
    
    return fig, ax, cr_dict


def dynesty_run_plot(stat_model):
    fig, ax = dyplot.runplot(stat_model.parameter_estimation_output)
    return fig, ax


def dynesty_trace_plot(stat_model):
    ndim = np.shape(stat_model.parameter_estimation_output.samples_u)[1]
    fig, ax = dyplot.traceplot(stat_model.parameter_estimation_output,
                                 truths=np.zeros(ndim),
                                 truth_color='black', show_titles=True, 
                                 trace_cmap='viridis', connect=True,
                                 connect_highlight=range(5))
    return fig, ax