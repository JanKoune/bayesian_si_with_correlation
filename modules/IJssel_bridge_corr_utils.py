# -*- coding: utf-8 -*-
"""
@author: Jan Koune
"""

'''
Collection of functions related to covariance, correlation and
likelihood evaluation for multivariate gaussian distributions. Contains
some other utility functions as well. 

TODO:
    * Organize this file better, add comments etc...
    * Compare results with prob_taralli for verification. Especially for the
    combined space and time covariance matrix calculation
'''

#%% ==========================================================================
# IMPORTS
# ============================================================================

import numpy as np
from sklearn import gaussian_process
from scipy.spatial.distance import pdist, squareform
import timeit
import math
from scipy.special import kv, gamma


def iid_loglike_2D_additive(y_res, std_meas, std_x, std_t = 1.0):
    """
    Efficient loglikelihood for a 2D i.i.d. MVN distribution assuming 
    multiplicative modeling uncertainty. Different uncertainty parameters can
    be specified for each of the two dimensions. 
    
    INPUT:
        y_obs: [S x T] array of residuals
        y_model: [S x T] array of model predictions
        std_meas: Measurement uncertainty std. dev.
        std_x: Modeling uncertainty std. dev. in the space dimension
        std_t: Modeling uncertainty std. dev. in the time dimension (Optional)
        
    """
    
    # Get grid size
    Nx, Nt = np.shape(y_res)
    
    # Convert to vector
    y_res = np.ravel(y_res)
    
    # Diagonals of space and time covariance matrices and measurement covariance
    # matrix
    cov_x = np.repeat(std_x ** 2, Nx)
    cov_t = np.repeat(std_t ** 2, Nt)
    cov_meas = np.repeat(std_meas ** 2, Nx * Nt)
    
    # Kronecker product of the space and time covariance. This should be equal
    # to the diaognal of the combined space and time covariance matrix.
    k_cov_xt = np.kron(cov_x, cov_t)
    cov_xt = k_cov_xt + cov_meas
    
    # Calculate determinant
    logdet_cov_xt = np.sum(np.log(cov_xt))

    # Loglikelihood
    return -Nx * Nt / 2 * np.log(2 * np.pi) - 0.5 * logdet_cov_xt - 0.5 * np.sum(y_res ** 2 * 1/cov_xt)


def iid_loglike_2D_multiplicative(y_res, y_func, std_meas, cov_x, cov_t = 1.0):
    """
    Efficient loglikelihood for a 2D i.i.d. MVN distribution assuming 
    multiplicative modeling uncertainty. Different uncertainty parameters can
    be specified for each of the two dimensions. 
    
    INPUT:
        y_obs: [S x T] array of residuals
        y_model: [S x T] array of model predictions
        std_meas: Measurement uncertainty std. dev.
        cov_x: Modeling uncertainty c.o.v. in the space dimension
        cov_t: Modeling uncertainty c.o.v. in the time dimension (Optional)
        
    """
    
    # Get grid size
    Nx, Nt = np.shape(y_res)
    
    # Convert to vectors
    y_res = np.ravel(y_res)
    y_func = np.ravel(y_func)
    
    # Diagonals of space and time covariance matrices before scaling by the
    # model output. Diagonal of the measurement covariance matrix
    corr_x = np.repeat(cov_x ** 2, Nx)
    corr_t = np.repeat(cov_t ** 2, Nt)
    cov_meas = np.repeat(std_meas ** 2, Nx * Nt)
    
    # Kronecker product of the space and time covariance. This should be equal
    # to the diaognal of the combined space and time covariance matrix.
    corr_xt = np.kron(corr_x, corr_t)
    kph_cov_xt = y_func ** 2 * corr_xt
    cov_xt = kph_cov_xt + cov_meas
    
    # Calculate determinant
    logdet_cov_xt = np.sum(np.log(cov_xt))

    # Loglikelihood
    return -Nx * Nt / 2 * np.log(2 * np.pi) - 0.5 * logdet_cov_xt - 0.5 * np.sum(y_res ** 2 * 1/cov_xt)


class kernels:
    """
    Kernel definitions and utility functions for performing Bayesian inference.
    Different kernel types are defined with the aim of minimizing the code needed
    to implement them in Bayesian inference. It should be as simple as:

        1. func_covariance = kernel(coordinates)
        2. covariance_matrix = func_covariance.forward(parameters)

    Input :
      coords: Must be a [npts x ndim] array of coordinates

    NOTES :
        * All hyperparameters are initialized as None so that the resulting covariance
    matrix will have obvious mistakes if the inputs to the covariance function
    are wrong during inference (e.g. if I forget to pass length_scale so length_scale
    is always equal to the initial value).


        * This implementation is based on the kernel module of sklearn, which is distributed
    under a BSD 3 clause license. It seems that this allows for modifying and
    redistributing the source code as long as the original copyright is included:

    BSD 3-Clause License

    Copyright (c) 2007-2021 The scikit-learn developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self, coords, std):
        self.coords = coords
        self.std = std
        self.ndim = np.shape(self.coords)[
            1
        ]  # If this gives an error then coords is probably not the right shape
        self.npts = np.shape(self.coords)[0]
        # Error if too many dimensions
        if self.ndim > 4:
            raise ValueError(f"Dimension can't be > 4 but is {self.ndim}")

        self.dist = squareform(pdist(coords, metric="euclidean"))
        self.params = None

    def forward(self, std, **params):
        # NOTE : Since some kernels are evaluated using sklearn and some within this
        # class, there is seperate self.params = params and self.set_params(**params)
        # The first one is to have a general list of params in the class that can
        # be checked if needed, the second actually passes parameters to the kernel.
        # NOTE : Since __call__ is slow on sklearn kernels, they will be replaced
        # by evaluations within the the self.evaluate() function
        self.std = std
        self.params = params
        self.std_diag_mx = np.diag(np.repeat(self.std, self.npts))
        self.set_params(**params)
        return self._evaluate()

    def corr(self, **params):
        self.params = params
        return self._call()

    def plot_kernel_function(self):
        # TODO
        return

    def time_self(self, std=1.0, nevals=100, **params):
        # Time self for nevals covariance evaluations
        t = timeit.Timer()
        t_eval = 0
        t_kernel_call = 0
        self.std = std
        self.std_diag_mx = np.diag(np.tile(std, self.npts))
        self.set_params(**params)

        for _i in range(nevals):
            t1 = t.timer()
            self._evaluate()
            t2 = t.timer()
            # val2 = self.kernel.__call__(self.dist)
            self._call()
            t3 = t.timer()

            # eval contains the kernel call! keep that in mind when timing
            t_kernel_call = t_kernel_call + (t3 - t2)
            t_eval = t_eval + (t2 - t1)

        return np.array([t_eval, t_kernel_call]) / nevals


class Independence(kernels):
    def __init__(self, coords, std=None):
        super().__init__(coords, std)
        self.param_list = []

    def set_params(self, **params):
        return

    def _call(self):
        return np.diag(np.ones(self.npts))

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class Correlated(kernels):
    def __init__(self, coords, std=None):
        super().__init__(coords, std)
        self.param_list = []

    def set_params(self, **params):
        return

    def _call(self):
        return np.ones((self.npts, self.npts))

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class RationalQuadratic(kernels):
    def __init__(self, coords, std=None, length_scale=None, alpha=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "alpha"]
        self.length_scale = length_scale
        self.alpha = alpha

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.alpha = params["alpha"]

    def _call(self):
        return (
            1 + self.dist ** 2 / (2 * self.alpha * self.length_scale ** 2)
        ) ** -self.alpha

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class ExpSineSquared(kernels):
    def __init__(self, coords, std=None, length_scale=None, periodicity=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "periodicity"]
        self.length_scale = length_scale
        self.periodicity = periodicity

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.periodicity = params["periodicity"]

    def _call(self):
        return np.exp(
            -2 * (np.sin(np.pi / self.periodicity * self.dist) / self.length_scale) ** 2
        )

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class RBF(kernels):
    def __init__(self, coords, std=None, length_scale=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale"]
        self.length_scale = length_scale

    def set_params(self, **params):
        self.length_scale = params["length_scale"]

    def _call(self):
        return np.exp(-self.dist ** 2 / (2 * self.length_scale ** 2))

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class Matern(kernels):
    def __init__(self, coords, std=None, length_scale=None, nu=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "nu"]
        self.length_scale = length_scale
        self.nu = nu

    def set_params(self, **params):
        self.nu = params["nu"]
        self.length_scale = params["length_scale"]

    def _call(self):
        dists = self.dist / self.length_scale
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists ** 2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)
        return K

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class DampedCosine(kernels):
    def __init__(self, coords, std=None, length_scale=None, wn=None):
        super().__init__(coords, std)
        self.length_scale = length_scale
        self.wn = wn
        self.kernel = gaussian_process.kernels.Kernel
        self.param_list = ["length_scale", "wn"]

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.wn = params["wn"]

    def _call(self):
        return np.exp(-self.dist / self.length_scale) * np.cos(self.dist * self.wn)

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class Exponential(kernels):
    def __init__(self, coords, std=None, length_scale=None):
        super().__init__(coords, std)
        self.length_scale = length_scale
        self.param_list = ["length_scale"]

    def set_params(self, **params):
        self.length_scale = params["length_scale"]

    def _call(self):
        return np.exp(-self.dist / (self.length_scale))

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class Cauchy(kernels):
    def __init__(self, coords, std=None, length_scale=None, exponent=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale", "exponent"]
        self.length_scale = length_scale
        self.exponent = exponent

    def set_params(self, **params):
        self.length_scale = params["length_scale"]
        self.exponent = params["exponent"]

    def _call(self):
        return (1 + (self.dist / self.length_scale) ** 2) ** -self.exponent

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)


class Gaussian(kernels):
    def __init__(self, coords, std=None, length_scale=None):
        super().__init__(coords, std)
        self.param_list = ["length_scale"]
        self.length_scale = length_scale

    def set_params(self, **params):
        self.length_scale = params["length_scale"]

    def _call(self):
        return np.exp(-((self.dist / self.length_scale) ** 2))

    def _evaluate(self):
        return np.matmul(np.matmul(self.std_diag_mx, self._call()), self.std_diag_mx)
