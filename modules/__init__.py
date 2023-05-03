# -*- coding: utf-8 -*-

# Silence julia Futurewarning and numpy deprecation warning related to the posterior
# plot. The Julia warning is printed every time there is console output
import warnings
import os
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import matplotlib
matplotlib.use("TkAgg")

#%% =================================================================
# PATHS
# ===================================================================

# Path to Julia excecutable. Must use forward slashes and end in bin/
path_julia = "C:/Users/ikoune/AppData/Local/Programs/Julia-1.8.3/bin/"

# These should not be changed.
ROOT_DIR = os.path.abspath(
                os.path.join(
                    os.path.dirname(
                        os.path.abspath(__file__)
                        )
                    , '..')
                )

path_sections = os.path.join(ROOT_DIR, "continuous_girder\\IJssel_bridge\\")
path_fem = os.path.join(ROOT_DIR, "continuous_girder\\")
path_likelihood = os.path.join(ROOT_DIR, "")
meas_path = os.path.join(ROOT_DIR, "measurements\\")
figures_path = os.path.join(ROOT_DIR, "figures\\")
data_path = os.path.join(ROOT_DIR, "data\\")
results_path = os.path.join(ROOT_DIR, "results\\")
paths = {
    'root': ROOT_DIR + '\\',
    'julia': path_julia,
    'sections': path_sections,
    'fem': path_fem,
    'likelihood': path_likelihood,
    'measurements': meas_path,
    'figures': figures_path,
    'data': data_path,
    'results': results_path,
    }
measurements_path = os.path.join(paths['measurements'], "measurements.csv")
