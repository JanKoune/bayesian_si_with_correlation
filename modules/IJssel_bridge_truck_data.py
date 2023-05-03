# -*- coding: utf-8 -*-
"""
@author: Jan Koune
"""
"""
Creates a dictionary with truck information for influence line calculation. Two
setups are included: the single sensor TNO measurement setup used initially
and the additional fugro sensors.

NOTES:
  * Taken from 'IJssel_influence_line_sciml.jl'
  * These dont necessarily give the same results as the corresponding .jl code

STRUCTURE:
    truck_data -> series -> lane -> data
"""

import numpy as np
from modules import paths

#%% ==========================================================================
# Assemble dictionaries
#
# Create a dictionary to store all the info. This will include the truck data
# for the two measurement series:
#   * TNO measurements from strain gauge at the second span
#   * Fugro measurements from multiple strain gauges
#
#
# STRUCTURE:
#     truck_data -> series -> lane -> data
#
# NOTES :
#   * For the fugro measurements, only the total truck weight and the axle distances
# are known to me. For now I will use the TNO left lane truck parameters for the
# since they are close enough (50.0 tons vs. 50.45 tons and one-two cm of difference
# in the axle distances)
# ============================================================================

keys_data = ["force", "axle_dist", "center_z", "wheel_z_dist"]
keys_lane = ["left", "right"]
keys_series = ["TNO", "fugro", "uniform"]

# Create the nested dictionary
truck_data = dict.fromkeys(keys_series)
for i, key_i in enumerate(truck_data.keys()):
    truck_data[key_i] = dict.fromkeys(keys_lane)
    for j, key_j in enumerate(truck_data[key_i]):
        truck_data[key_i][key_j] = dict.fromkeys(keys_data)

#%% =========================================================================
# TRUCK DATA
# ===========================================================================

right_truck_force = np.array(
    [
        [57.518 / 2, 57.518 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
    ]
)
right_truck_axle_dist_x = np.array([1.94, 2.09, 1.345, 1.25])
right_truck_center_z = 5.700 / 2 - 3.625 / 2
right_truck_wheel_z_dist = 2.1500
right_truck_num_axle = np.shape(right_truck_force)[0]

left_truck_force = np.array(
    [
        [58.86 / 2, 58.86 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
    ]
)
left_truck_axle_dist_x = np.array([2.0, 1.82, 1.82, 1.82])
left_truck_center_z = 5.700 / 2 + 3.625 / 2
left_truck_wheel_z_dist = right_truck_wheel_z_dist
left_truck_num_axle = np.shape(left_truck_force)[0]

right_ax_flatlist = np.cumsum(right_truck_axle_dist_x)
right_ax_flatlist = np.insert(right_ax_flatlist, 0, 0)

right_truck_x = np.repeat(right_ax_flatlist, 2)
right_truck_z = np.hstack(
    (
        right_truck_center_z
        - right_truck_wheel_z_dist / 2 * np.ones(right_truck_num_axle),
        right_truck_center_z
        + right_truck_wheel_z_dist / 2 * np.ones(right_truck_num_axle),
    )
)

left_ax_flatlist = np.cumsum(left_truck_axle_dist_x)
left_ax_flatlist = np.insert(left_ax_flatlist, 0, 0)

left_truck_x = np.repeat(left_ax_flatlist, 2)
left_truck_z = np.hstack(
    (
        left_truck_center_z
        - left_truck_wheel_z_dist / 2 * np.ones(left_truck_num_axle),
        left_truck_center_z
        + left_truck_wheel_z_dist / 2 * np.ones(left_truck_num_axle),
    )
)

# Assuming that z = 0 at the center of the bridge
z_truck_r_wheel_r = right_truck_center_z - right_truck_wheel_z_dist / 2
z_truck_r_wheel_l = right_truck_center_z + right_truck_wheel_z_dist / 2
z_truck_l_wheel_r = left_truck_center_z - left_truck_wheel_z_dist / 2
z_truck_l_wheel_l = left_truck_center_z + left_truck_wheel_z_dist / 2

# Fugro truck forces: the total weight is given as 50.420 tonnes.
# The force is calculated as Wtot * g * 0.12 for the first axle and
# Wtot * g * 0.22 for the other axles
truck_axle_dist_x_fugro = np.array([2.06, 1.83, 1.82, 1.82])
truck_force_fugro = np.array(
    [
        [59.35 / 2, 59.35 / 2],
        [108.82 / 2, 108.82 / 2],
        [108.82 / 2, 108.82 / 2],
        [108.82 / 2, 108.82 / 2],
        [108.82 / 2, 108.82 / 2],
    ]
)

# Uniform load
right_truck_axle_dist_x = np.array([1.94, 2.09, 1.345, 1.25])
left_truck_axle_dist_x = np.array([2.0, 1.82, 1.82, 1.82])
right_dist_tot = np.sum(right_truck_axle_dist_x)
left_dist_tot = np.sum(left_truck_axle_dist_x)
left_truck_force_single = np.array(
    [
        [58.86 / 2, 58.86 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
        [107.91 / 2, 107.91 / 2],
    ]
)
right_truck_force_single = np.array(
    [
        [57.518 / 2, 57.518 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
        [105.45 / 2, 105.45 / 2],
    ]
)
left_force_tot = np.sum(left_truck_force_single, axis=0)
right_force_tot = np.sum(right_truck_force_single, axis=0)

Nloads = 10
left_force_uniform = np.tile(left_force_tot / Nloads, (Nloads, 1))
right_force_uniform = np.tile(right_force_tot / Nloads, (Nloads, 1))
left_truck_axle_dist_uniform = np.repeat(left_dist_tot / (Nloads-1), Nloads-1)
right_truck_axle_dist_uniform = np.repeat(right_dist_tot / (Nloads-1), Nloads-1)


#%% =========================================================================
# Pass the values to the dictionary
# ===========================================================================

truck_data["TNO"]["left"]["force"] = left_truck_force
truck_data["TNO"]["left"]["axle_dist"] = left_truck_axle_dist_x
truck_data["TNO"]["left"]["center_z"] = left_truck_center_z
truck_data["TNO"]["left"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["TNO"]["left"]["z_wheel_r"] = z_truck_l_wheel_r
truck_data["TNO"]["left"]["z_wheel_l"] = z_truck_l_wheel_l

truck_data["TNO"]["right"]["force"] = right_truck_force
truck_data["TNO"]["right"]["axle_dist"] = right_truck_axle_dist_x
truck_data["TNO"]["right"]["center_z"] = right_truck_center_z
truck_data["TNO"]["right"]["wheel_z_dist"] = right_truck_wheel_z_dist
truck_data["TNO"]["right"]["z_wheel_r"] = z_truck_r_wheel_r
truck_data["TNO"]["right"]["z_wheel_l"] = z_truck_r_wheel_l

truck_data["fugro"]["left"]["force"] = truck_force_fugro
truck_data["fugro"]["left"]["axle_dist"] = truck_axle_dist_x_fugro
truck_data["fugro"]["left"]["center_z"] = left_truck_center_z
truck_data["fugro"]["left"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["fugro"]["left"]["z_wheel_r"] = z_truck_l_wheel_r
truck_data["fugro"]["left"]["z_wheel_l"] = z_truck_l_wheel_l

truck_data["fugro"]["right"]["force"] = truck_force_fugro
truck_data["fugro"]["right"]["axle_dist"] = truck_axle_dist_x_fugro
truck_data["fugro"]["right"]["center_z"] = right_truck_center_z
truck_data["fugro"]["right"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["fugro"]["right"]["z_wheel_r"] = z_truck_r_wheel_r
truck_data["fugro"]["right"]["z_wheel_l"] = z_truck_r_wheel_l

truck_data["uniform"]["left"]["force"] = left_force_uniform
truck_data["uniform"]["left"]["axle_dist"] = left_truck_axle_dist_uniform
truck_data["uniform"]["left"]["center_z"] = left_truck_center_z
truck_data["uniform"]["left"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["uniform"]["left"]["z_wheel_r"] = z_truck_l_wheel_r
truck_data["uniform"]["left"]["z_wheel_l"] = z_truck_l_wheel_l

truck_data["uniform"]["right"]["force"] = right_force_uniform
truck_data["uniform"]["right"]["axle_dist"] = right_truck_axle_dist_uniform
truck_data["uniform"]["right"]["center_z"] = right_truck_center_z
truck_data["uniform"]["right"]["wheel_z_dist"] = left_truck_wheel_z_dist
truck_data["uniform"]["right"]["z_wheel_r"] = z_truck_r_wheel_r
truck_data["uniform"]["right"]["z_wheel_l"] = z_truck_r_wheel_l
