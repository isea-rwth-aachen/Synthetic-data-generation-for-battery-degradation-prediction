# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 21:36:19 2022

@author: HarshvardhanSamsukha
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Define bacon-watts formula
def bacon_watts_model(x, alpha0, alpha1, alpha2, x1):
    ''' Equation of bw_model'''
    return alpha0 + alpha1*(x - x1) + alpha2*(x - x1)*np.tanh((x - x1) / 1e-8)


def fit_bacon_watts(y, p0):
    ''' 
    Function to fit Bacon-Watts model to identify knee-point
    in capacity fade data    
    
    Args:
    - capacity fade data (list): cycle-to-cycle evolution of Qd capacity
    - p0 (list): initial parameter values for Bacon-Watts model
    Returns:
    - popt (int): fitted parameters
    - confint (list): 95% confidence interval for fitted knee-point
    
    '''         
    # Define array of cycles
    x = np.arange(len(y)) + 1
    # Fit bacon-watts
    popt, pcov = curve_fit(bacon_watts_model, x, y, p0=p0)
    confint = [popt[3] - 1.96 * np.diag(pcov)[3], 
               popt[3] + 1.96 * np.diag(pcov)[3]]
    return popt, confint

# Knee-point extraction function
def get_knee_point(data_list, i):
    p0_actual_tr = [1, -1e-4, -1e-4, len(data_list[i])*.7]
    popt_actual_tr, confint_actual = fit_bacon_watts(
        data_list[i], p0_actual_tr
    )
    return int(popt_actual_tr[3])

# EOL extraction function
def get_eol_point(eol_def, data_list, cycle_list, i):
    y_eol = eol_def * data_list[i][0]
    f = interp1d(data_list[i], cycle_list[i])
    x_eol = f(y_eol)
    return int(x_eol)

