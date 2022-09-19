# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 21:41:26 2022

@author: HarshvardhanSamsukha
"""

import matplotlib.pyplot as plt

def plot_data(
    data_list, cycle_list,
    synthetic_data_list, synthetic_cycle_list,
    color_real='blue', color_syn='red',
):
    plt.figure(figsize=(10,8))
    for i in range(len(synthetic_data_list)):
        plt.plot(
            synthetic_cycle_list[i], synthetic_data_list[i], color=color_syn
        )
    plt.plot(
        synthetic_cycle_list[i], synthetic_data_list[i],
        label='Synthetic data', color=color_syn
    )
    for i in range(len(data_list)):
        plt.plot(cycle_list[i], data_list[i], color=color_real)
    plt.plot(cycle_list[i], data_list[i], label='Real data', color=color_real)
    plt.legend(loc='best')
    plt.show()

def plot_individual_data(data_list, cycle_list, title, color='blue'):
    plt.figure(figsize=(8,6))
    for i in range(len(data_list)):
        plt.plot(cycle_list[i], data_list[i], color=color)
    plt.plot(cycle_list[i], data_list[i], color=color)
    plt.xlabel('Cycles')
    plt.ylabel('Capacity (Ah)')
    plt.title(title)
    plt.show()
    