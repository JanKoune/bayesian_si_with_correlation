"""
@author: Jan Koune
"""

"""
Model file for the IJsselbridge case study. The parts of the code that are called
most often in the main file were moved to their own seperate files. In this case
we define the class 'IJssel_bridge_model' which should make it easier and faster
to test, fix and add to the model.

NOTES : 
    * Not all precalculated stress influence lines are updated when the support
    stiffnesses are updated! Only the one for which the influence line is calculated
    * It is assumed that the spring stiffnesses are given in MN/m

TODOs : 
    * Make a detailed comparison with the julia code to ensure we get the same
    results for the same inputs. 
    * Add comments for the functions
    * Add type hints
    * Make sure supports are added to the correct nodes
    * Truck loads are not in agreement with the TNO report. Neither are the lanes
    * Replace all the if statements in il_stress_truckload with dicts for the 
    corresponding quantities
    * Make sure that the truck forces are applied correctly. Specifically, that
    the loads are applied to the nodes in the correct order and that the 
    point load vector is not flipped
    
This will be changed to use Betti's theorem for calculation of the influence line.
This is a lot faster than the current method, which requires a solve for each
point of the influence line. Instead using Bettis theorem we calculate the entire
influence line at the sensor location for a moving point load. We can then get the
i.l. for multiple point loads:
    - for forces f1, f2,.. at positions x1, x2, ... we get the value of the influence
    line by superposition of the values for f1, f2, ...
    - There should be a way to do this by constructing a vector for each of the 
    individual influence lines il1, il2, ... and then summing them all together.
    This would be preferable to looping over all the possible load positions.
"""

import os
import timeit
from copy import copy, deepcopy

import matplotlib.pyplot as plt
#%% ==========================================================================
# IMPORTS
# ============================================================================
import numpy as np
from numba import jit
from scipy.linalg import cho_factor, cho_solve

from modules.IJssel_bridge_truck_data import truck_data
from modules import paths
from julia import Julia

julia_bin_path = paths["julia"]
libpaths = ["libgomp-1.dll", "libpcre2-8-0.dll"]
try:
    jl = Julia(runtime=julia_bin_path + "julia.exe")
except:
    jl = Julia(runtime=julia_bin_path + "julia.exe")
jl.eval("using Libdl")
for libname in libpaths:
    libdl_eval_string = "Libdl.dlopen(" + '"' + julia_bin_path + libname + '"' ")"
    jl.eval(libdl_eval_string)
jl.eval("using LinearAlgebra")
from julia import Main




#%% ==========================================================================
# FEM SOLVE FUNCTIONS
# ============================================================================
def lateral_load_func(x, z, c):
    z = np.asarray(z)
    
    # If c is a single value, assumes its a linear coefficient. otherwise its 
    # assumed to be the n coefficients of a polynomial
    c = np.atleast_1d(c)
    if (not np.shape(c)) or (len(c) == 1):
        # TODO : This is a bad fix. Find a way to keep the input consistent
        # (this is to fix the case where c can be either scalar or array of 1)
        if not np.isscalar(c):
            return c[0]*(z - 2.85) + 0.5
        else:
            return c*(z - 2.85) + 0.5
    else:
        return np.polyval(c, z - 2.85)

@jit(nopython = True)
def find_nearest(array, values):
    # By default in case of multiple equal values the first is returned 
    # (see numpy documentation for argmin())
    array = np.asarray(array)
    values = np.atleast_1d(np.asarray(values))
    idx = np.zeros(len(values), dtype = np.int32)
    for i, val in enumerate(values):
        idx[i] = np.abs(array - val).argmin()
    return idx

def find_all(array, values):
    array = np.asarray(array)
    values = np.atleast_1d(np.asarray(values))
    idx = []
    for i, val in enumerate(values):
        idx.append(np.where(array == val))
    return np.ravel(idx)

# @jit(nopython = True)
def betti(nnode, Ks, W_sensor, hinge_node, hinge_left_node, fs, idx_keep):
    
    u = np.zeros(2 * nnode)
    
    # Note: Given equal Ks and fs, both the cholesky and conventional solve
    # will give slightly different results than the corresponding Julia
    # solve: us = Ks \ fs.
    # TODO: Test this more. The difference seems too large to be within the
    #   expected error for a linear system solve.

    # Factorize and solve with scipy
    us = cho_solve(cho_factor(Ks), fs, check_finite = False)
    # us = np.linalg.solve(Ks, fs)

    # Factorize and solve with numpy
    # cho_factor_np = np.linalg.cholesky(Ks)
    # y = np.linalg.solve(cho_factor_np, fs)
    # us = np.linalg.solve(cho_factor_np.T, y)

    u[idx_keep] = us
    r_hinge = u[hinge_left_node * 2 + 1] - u[hinge_node * 2 + 1]

    # scale to unit hinge rotation
    m_b_1 = 1 / r_hinge * u[0::4]
    m_b_2 = 1 / r_hinge * u[2::4]

    return np.asarray([m_b_1, m_b_2])/W_sensor

def solve_inf_line(m_b, axle_pos_x, node_xs, loads):
    stress_1 = loads[:] * np.interp(axle_pos_x[:], node_xs, m_b[0])
    stress_2 = loads[:] * np.interp(axle_pos_x[:], node_xs, m_b[1])
    return np.sum(stress_1, axis = 1), np.sum(stress_2, axis = 1)


#%% =========================================================================
# IMPORT JULIA FUNCTIONS

julia_path_sections = paths['sections']
julia_path_fem = paths['fem']
measurements_path = os.path.join(paths['measurements'], "measurements.csv")

Main.include(os.path.join(julia_path_sections, "girder_sections.jl"))
Main.include(os.path.join(julia_path_fem, "FEM_girder.jl"))
Main.include(os.path.join(julia_path_fem, "FEM_utils.jl"))
Main.include(os.path.join(julia_path_fem, "utils.jl"))

#%% ==================================================================
# INITIALIZATION
# ====================================================================
class IJssel_bridge_model:
    def __init__(self, sensor_position, E, max_elem_length, additional_node_pos = [], truck_load = 'TNO'):
        
        self.sensor_position = sensor_position
        self.E = E
        self.max_elem_length = max_elem_length
        self.truck_load = truck_load
        
        # Apply hinges for Betti influence line calculation.
        self.hinge_left_pos = self.sensor_position - 0.003
        
        # Sensor and hinge additional node positions
        self.additional_node_positions = np.array(
            np.append([self.hinge_left_pos, self.sensor_position],
                      additional_node_pos
                      )
            ) * 1e3
        
        #%% ==================================================================
        # STRUCTURE PROPERTIES FROM JULIA MODEL
        # ====================================================================
        
        # Get structural properties
        self.section_properties = (
            Main.girder_sections(
                max_elem_length=self.max_elem_length,
                additional_node_positions=self.additional_node_positions,
                consider_K_braces = True
            )
        )
        
        # Copy from dict
        self.support_xs = self.section_properties["support_xs"]/1e3
        self.node_xs = self.section_properties["node_xs"]/1e3
        self.K_brace_xs = self.section_properties["K_brace_xs"]/1e3
        self.elem_c_ys = self.section_properties["elem_c_ys"]/1e3
        self.elem_h_ys = self.section_properties["elem_h_ys"]/1e3
        self.elem_I_zs = self.section_properties["elem_I_zs"]/1e12
        
        # Element stiffness
        self.elem_EIs = self.elem_I_zs * self.E
        self.W_bot_temp = self.elem_I_zs / (self.elem_h_ys - self.elem_c_ys)
        self.elem_W_bottom = np.repeat(self.W_bot_temp, 2)
        self.nelems = len(self.elem_EIs)
        
        # Assemble K
        self.Ks0, self.nodes, self.idx_keep, lin_idx_springs = Main.fem_general_twin_girder(self.node_xs,
                                                                                                 self.elem_EIs,
                                                                                                 self.support_xs,
                                                                                                 spring_positions = self.K_brace_xs,
                                                                                                 spring_stiffnesses = np.zeros(np.shape(self.K_brace_xs)),
                                                                                                 left_hinge_positions=[sensor_position])
        
        # NOTE : This gives the closest node. It does not ensure that sensor_pos == node_pos
        self.sensor_node = find_nearest(self.nodes[:,0], sensor_position)[0]
        self.hinge_node = self.sensor_node
        self.hinge_left_node = find_nearest(self.nodes[:,0], self.hinge_left_pos)[0]
        self.W_sensor = self.elem_W_bottom[self.sensor_node]
        
        # Convert the flat, 1-indexed indices from Julia into numpy indices
        # TODO : Also make sure to check if the correct numbering is used
        # (Fortran vs. C numbering)
        # QUESTION : Why is len(lin_idx_spring_off_diag) != len(K_brace_xs) ???
        K_shape = np.shape(self.Ks0)
        lin_idx_spring_off_diag = np.asarray(lin_idx_springs["off_diag"]) - 1
        lin_idx_spring_diag = np.asarray(lin_idx_springs["diag"]) - 1
        self.idx_spring_off_diag = np.unravel_index(lin_idx_spring_off_diag, K_shape)
        self.idx_spring_diag = np.unravel_index(lin_idx_spring_diag, K_shape)

        # TODO : Use sparse matrices for Ks to improve performance.
        self.nnode = np.shape(self.nodes)[0]
        self.ndof = np.shape(self.Ks0)[0]
        self.support_nodes = find_all(self.nodes[:,0], self.support_xs)
        
        # Boolean masking for applying rotational stiffnesses to Ks
        self.Ks_stiff_mask = np.asarray(np.zeros(2*self.nnode), dtype = bool)
        self.Ks_stiff_mask[self.support_nodes*2 + 1] = 1
        self.Ks_stiff_mask = self.Ks_stiff_mask[self.idx_keep]
        
        # Copy and factorize Ks. Ks and K are related as follows:
        #       # Ks = K[idx_keep][:, idx_keep]
        # self.Ks = deepcopy(self.Ks0)
        
        # Precalculate unit load pair for betti theorem
        f = np.zeros(self.nnode * 2)
        f[self.hinge_left_node * 2 + 1] = 1e+6
        f[self.hinge_node * 2 + 1] = -1e+6
        self.fs = f[self.idx_keep]

        # # Precalculate unit influence line
        # self.unit_il= betti(self.nnode,
        #                      self.Ks_factor,
        #                      self.W_sensor,
        #                      self.sensor_node,
        #                      self.hinge_left_node,
        #                      self.fs,
        #                      self.idx_keep)

        # Precalculate unit influence line
        self.unit_il= betti(self.nnode,
                             self.Ks0,
                             self.W_sensor,
                             self.sensor_node,
                             self.hinge_left_node,
                             self.fs,
                             self.idx_keep)


        #%% ==================================================================
        # TRUCK DATA
        #
        # The truck data corresponding to 'TNO' or 'fugro' measurements is 
        # loaded. Default is 'TNO'.
        # ====================================================================
        
        # Wheel z positions 
        z_truck_r_wheel_r = truck_data[truck_load]['right']['z_wheel_r']
        z_truck_r_wheel_l = truck_data[truck_load]['right']['z_wheel_l']
        z_truck_l_wheel_r = truck_data[truck_load]['left']['z_wheel_r']
        z_truck_l_wheel_l = truck_data[truck_load]['left']['z_wheel_l']
        
        # Forces
        right_truck_force = truck_data[truck_load]['right']['force']
        left_truck_force = truck_data[truck_load]['left']['force']
        
        # Axle distances
        right_truck_axle_dist_x = truck_data[truck_load]['right']['axle_dist']
        left_truck_axle_dist_x = truck_data[truck_load]['left']['axle_dist']
        
        # Create objects holding the parameter values for left, right and double
        # case. This is to avoid the if-else statements that were used previously.
        self.truck_z = {
            'left': [z_truck_l_wheel_r, z_truck_l_wheel_l],
            'right': [z_truck_r_wheel_r, z_truck_r_wheel_l],
            'double': [z_truck_r_wheel_r, z_truck_r_wheel_l, z_truck_l_wheel_r, z_truck_l_wheel_l]
            }
        self.truck_force = {
            'left': np.transpose(left_truck_force),
            'right': np.transpose(right_truck_force),
            'double': np.vstack((np.transpose(right_truck_force), np.transpose(left_truck_force)))
            }
        self.truck_axle_dist = {
            'left': np.tile(left_truck_axle_dist_x, (2,1)),
            'right': np.tile(right_truck_axle_dist_x, (2,1)),
            'double': np.vstack(
                    (np.tile(right_truck_axle_dist_x, (2,1)),
                     np.tile(left_truck_axle_dist_x, (2,1))))
            }
        
        #%% ==================================================================
        # PRECALCULATE LOAD PATHS AND INFLUENCE LINES
        #
        # NOTES:
        #   * This is not optimized for speed. This should not be a problem 
        #   since this only runs once on model initialization. 
        # ====================================================================
        
        # Dictionaries to store precalculation output
        all_paths = ['right', 'left', 'double']
        self._truck_load = dict.fromkeys(all_paths)
        self.axle_pos_x = dict.fromkeys(all_paths)
        self.loaded_nodes = dict.fromkeys(all_paths)
        self.loads = dict.fromkeys(all_paths)
        
        for i, lane in enumerate(all_paths):
            
            # Assemble the vectors of loads and axle distances
            loads = np.atleast_2d(self.truck_force[lane])
            axle_dist_x = np.atleast_2d(self.truck_axle_dist[lane])
            ax_dist = np.cumsum(axle_dist_x[0,:])
            load_i = loads[0,:]
            
            ax_pos = []
            l_node = []
            load = []
            for j in range(len(self.node_xs)):
                # For each load position sum the corresponding moments from the il and
                # calculate the stress at the sensor position
                curr_pos = np.append(self.node_xs[j], self.node_xs[j] - ax_dist)
                ax_pos.append(curr_pos)

                # TODO: this can be improved. Instead of finding the closest node,
                #   do linear interpolation of the unit il for the actual position

                l_node.append(find_nearest(self.node_xs, curr_pos))
                
                # Make load zero for points that have not fully entererd the st
                # ructure yet
                temp_load = copy(load_i)
                temp_load[curr_pos < self.node_xs[0]] = 0
                load.append(temp_load)

            # Append to global list
            self.axle_pos_x[lane] = ax_pos
            self.loaded_nodes[lane] = l_node
            self.loads[lane] = load
            self._truck_load[lane] = np.tile(
                solve_inf_line(self.unit_il,
                               self.axle_pos_x[lane],
                               self.node_xs,
                               self.loads[lane]),
                (len(self.truck_z[lane]), 1))
        
    #%% ======================================================================
    # DEFINE FUNCTIONS
    # ========================================================================

    def il_stress_truckload(self, c, lane: str, Kr = None, Kv = None):
        # Check if stiffnesses have been updated, if yes, update the il
        # The brige supports 1 - 6 are labeled: F, G, H, J, K, L. Only the 
        # first 4 are considered, based on the TNO report.
        if (lane == 'double') or (lane == 'Double'):
            raise ValueError("Not implemented")
        
        if (Kr is not None) or (Kv is not None):

            # Add new stiffnesses to stiffness matrix
            self.Ks = copy(self.Ks0)
            self.Ks[self.Ks_stiff_mask, self.Ks_stiff_mask] += np.asarray(Kr)
            self.Ks[self.idx_spring_diag] += Kv
            self.Ks[self.idx_spring_off_diag] -= Kv

            # Calculate new influence line
            self.unit_il = betti(self.nnode,
                                 self.Ks,
                                 self.W_sensor,
                                 self.sensor_node, 
                                 self.hinge_left_node,
                                 self.fs,
                                 self.idx_keep)      
        # The returned stress arrays contain two stress influence lines for each
        # truck z value. lines [0::2] correspond to the right girder stress and
        # lines [1::2] correspond to the left girder stress
        
        stress = np.zeros(len(self.node_xs))

        self._truck_load[lane] = np.tile(
            solve_inf_line(self.unit_il,
                           self.axle_pos_x[lane],
                           self.node_xs,
                           self.loads[lane]),
            (len(self.truck_z[lane]), 1))

        
        for i, z in enumerate(self.truck_z[lane]):
            stress += lateral_load_func(0, z, c)*self._truck_load[lane][2*i, :]
            stress += lateral_load_func(0, z, -c)*self._truck_load[lane][2*i+1, :]

        return stress / 1000 # Return stress in MPa
