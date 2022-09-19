# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 10:38:29 2022

@author: h.samsukha
"""

import numpy as np
import pandas as pd

from functions.data_preprocessing import cap_data_list_creator, cycle_list_creator
from functions.synthetic_data_generator import random_synthetic_data
from functions.plotting import plot_data, plot_individual_data

# %% Load data
raw_data = pd.read_excel('data/NASA_Battery_Data_Set.xlsx', engine='openpyxl')

# %% Extracting capacity data from dataframe and storing in a list
data_list = cap_data_list_creator(raw_data)
# Creating cycle data assuming one cycle for each capacity point
cycle_list = cycle_list_creator(data_list)
selected_cells = np.array([i for i in range(len(data_list))])  # All cells selected

# %% Generating synthetic data, one curve per cell in original data
synthetic_data_list = random_synthetic_data(
    selected_cells, data_list, cycle_list, n_syn_curves=20, noise_sigma=1e-5,
    offset_param_bounds=[-0.02, 0.02],
    slope_param_bounds=[0, 0],
    elong_param_bounds=[0.75, 1.25]
)
# Cycles data for the synthetic data, assuming one cycle per capacity point
synthetic_cycle_list = cycle_list_creator(synthetic_data_list)

# %% Plotting
plot_data(
    data_list, cycle_list,
    synthetic_data_list, synthetic_cycle_list,
)
plot_individual_data(data_list, cycle_list, 'Real data', color='blue')
plot_individual_data(
    synthetic_data_list, synthetic_cycle_list,
    'Synthetic data', color='red'
)
