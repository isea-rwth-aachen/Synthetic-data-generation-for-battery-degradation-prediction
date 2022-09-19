# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:17:09 2022

@author: HarshvardhanSamsukha
"""

import numpy as np


def cap_data_list_creator(raw_data):
    # Drop any columns having cycle numbers
    cols = [c for c in raw_data.columns if c.lower()[:8] == 'capacity']
    raw_data = raw_data[cols]
    data_list = []
    for column in raw_data.columns:
        a = raw_data[column].dropna().values
        data_list.append(a)
    return data_list

def cycle_list_creator(data_list, freq=1):
    cycle_list = []
    for cell in data_list:
        a = np.linspace(0, len(cell)*freq, len(cell), endpoint=False)
        cycle_list.append(a)
    return cycle_list
