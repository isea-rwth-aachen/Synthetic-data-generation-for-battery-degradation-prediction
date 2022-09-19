# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:02:06 2022

@author: HarshvardhanSamsukha
"""

import numpy as np


def random_synthetic_data(
    selected_cells,
    data_list,
    cycle_list,
    n_syn_curves,
    noise_sigma=1e-5,
    offset_param_bounds=[-1e-2, 1e-2],
    slope_param_bounds=[1,1],
    elong_param_bounds=[1,1]
):
    """
    Generates completely random synthetic data for different cells within
    the set bounds of synthetic data parameters - offset, slope, elongation.
    Params:
        selected_cells: Array of indices of the cells within data_list for 
                        which synthetic data is needed.
        data_list: Capacity/time-series data of the cells.
        cycle_list: Cycle-number data of the cells.
        n_syn_curves: Number of synthetic data to be generated.
        noise_sigma: Standard deviation of noise level desired in the
                     synthetic data.
        offset_param_bounds: Bounds of offset parameter [lower, upper]
        slope_param_bounds: Bounds of slope parameter [lower, upper]
        elong_param_bounds: Bounds of elongation parameter [lower, upper]
    
    Returns:
        List of synthetic capacity/time-series data
    """
    synthetic_capacity_data = []   # Final list of capacity
    multiple = (n_syn_curves // len(data_list)) + 1
    for i in range(multiple):
        selected_cells = selected_cells.astype(int)
        # Collecting cycle data of selected cells in a list
        selected_cell_cycle_data = []
        for i in selected_cells:
            selected_cell_cycle_data.append(cycle_list[i])

        # Mapping the cell numbers and their corresponding cycle data in a
        # dictionary for ease of tracking of data
        total_selected_cells = {}
        for i in range(len(selected_cells)):
            total_selected_cells[selected_cells[i]]=selected_cell_cycle_data[i]

        # Selecting elongation parameter randomly (within the set bounds)
        # for each cell's cycle data
        synthetic_cycle_data = []
        synthetic_cell_number = []
        for i in total_selected_cells.keys():
            # Choosing a random elongation parameter value from the bounds
            elong_param = np.random.uniform(elong_param_bounds[0],
                                            elong_param_bounds[1])
            cycle_data = [j for j in total_selected_cells[i]]
            # Creating elongation parameter-array for multiplication with
            # cycle data
            elongation  = np.linspace(1, elong_param, len(cycle_data),
                                      endpoint=True)
            # Multiplying with cycle data
            cycle_data = np.multiply(cycle_data, elongation)
            # Avoid cycle data if elongation factor selected = 0.
            if all(v==0 for v in cycle_data) == False:
                synthetic_cycle_data.append(cycle_data)
                synthetic_cell_number.append(i)

        # Mapping the synthetic cell numbers and their corresponding
        # elongated cycle data in a dictionary for ease of tracking data
        total_synthetic_cycle_data = {}
        for i in range(len(synthetic_cell_number)):
            total_synthetic_cycle_data[
                synthetic_cell_number[i]
            ] = synthetic_cycle_data[i]

        # Interpolating (cycle = integer) to get the final synthetic data list
        for i in total_synthetic_cycle_data.keys():
            offset = np.random.uniform(offset_param_bounds[0],
                                       offset_param_bounds[1])
            x_max = int(max(total_synthetic_cycle_data[i]))
            # Interpolating integer values for cycle numbers of synthetic data
            x_new = np.linspace(1, x_max, num=x_max, endpoint=True)
            y_old = data_list[i]
            x_old = total_synthetic_cycle_data[i]
            # Interpolating capacity data points at interger cycle numbers
            y_new = np.interp(x_new, x_old, y_old)
            # Adding offset and slope parameters,
            # noise and other changes to capacity data
            y_new = y_new + offset
            slope_param = np.random.uniform(slope_param_bounds[0],
                                            slope_param_bounds[1])
            slope = np.linspace(0, slope_param, len(y_new), endpoint=True)
            y_new = np.add(y_new, slope)
            noise = np.random.normal(0, noise_sigma, np.shape(y_new))
            y_new = y_new+noise
            synthetic_capacity_data.append(y_new)
    # Removing extra synthetic curves to generate n_syn_curves
    del synthetic_capacity_data[-(multiple*len(data_list)-n_syn_curves):]
    return synthetic_capacity_data


