# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:19:53 2022

@author: harsh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from functions.data_preprocessing import (
    cap_data_list_creator, cycle_list_creator
)
from functions.synthetic_data_generator import random_synthetic_data
from functions.kneepoint_eol_detection import get_knee_point, get_eol_point
from functions.models import cnn_model, gpr_model

# Directories
savepath = 'results/'
# CNN model savepaths
model_savepath = 'results/temp/model.h5'
ckpt_val_loss  = 'results/temp/val_loss.h5'

#%% Data Loading and preprocessing
raw_data = pd.read_excel('data/RWTH_Dataset.xlsx', engine='openpyxl')
# Extracting capacity data from dataframe and storing in a list
data_list_o = cap_data_list_creator(raw_data)
frequency = 1
# Creating cycle data assuming one cycle for each capacity point
cycle_list_o = cycle_list_creator(data_list_o, freq=frequency)

#%% Model performance for different sets of randomly selected training data
# Loop parameters
n_training_samples = 3 # No. of real data used
n_syn_curves = 30 # No. of synthetic data to be generated from real data
n_runs = 5  # No. of runs (iterations)
model = 'cnn'  # 'gpr' or 'cnn'
target = 'eol' # 'kp' or 'eol'
eol_def = 0.8  # EOL of cell as a percentage of initial capacity
# Synthetic data parameters
offset_param_bounds= [-0.02, 0.02]
slope_param_bounds = [0, 0]
elong_param_bounds = [0.75, 1.25]

#%% Looping through runs = n_runs
# Total errors in all iterations to be stored in the following lists
list_pred_errors_cycles = []
list_pred_errors_percent = []
# Segregating test cells from total list of cells based on cell number
total_cell_numbers = [i for i in range(len(data_list_o))]
# The following were selected randomly and fixed for RWTH dataset
test_cell_numbers = [3, 15, 19, 34, 29, 45, 11]
rem_cell_numbers = [
    cell for cell in total_cell_numbers if cell not in test_cell_numbers
]

for i in range(n_runs):
    print("\nIteration", i+1, "\n")
    # Extract cells for training
    train_cell_numbers = np.array(
        random.sample(rem_cell_numbers, k = n_training_samples)
    ).astype(int)
    train_cells = np.array(train_cell_numbers.copy()).astype(int)
    data_list_train = []
    cycle_list_train= []
    for i in train_cells:
        data_list_train.append(data_list_o[i])
        cycle_list_train.append(cycle_list_o[i])
    train_cells = np.array([i for i in range(len(data_list_train))])
    # Selecting cell numbers for which synthetic data needs to be generated
    selected_cells = np.array([i for i in range(len(data_list_train))]) # All cells in data_list_train
    
    # Randomly generated synthetic data
    synthetic_data_list = random_synthetic_data(
        selected_cells, data_list_train, cycle_list_train,
        n_syn_curves = n_syn_curves,
        noise_sigma=1e-5,
        offset_param_bounds= offset_param_bounds,
        slope_param_bounds = slope_param_bounds,
        elong_param_bounds = elong_param_bounds
    )
    synthetic_cycle_list = cycle_list_creator(synthetic_data_list)

    # Combining the synthetic data with the original (smooth) data
    data_list_train.extend(synthetic_data_list)
    cycle_list_train.extend(synthetic_cycle_list)
    # Errors per iteration stored in following lists
    avg_percent_error_list = []
    avg_error_list = []

    # Prepping the data on which prediction needs to be made
    pred_from_cycle_list = [
        50, 100, 150, 200, 250, 300, 350, 400, 450,
        500, 550, 600, 650, 700, 750, 800, 850, 900
    ]

    for pred_from_cycle in pred_from_cycle_list:
        freq = 2    # Sampling every 'n'th point of data for training models
        # Train data
        X_train = []
        y_train = []
        for i in range(len(data_list_train)):
            # Inputs
            X_train.append(data_list_train[i][:pred_from_cycle:freq])
            # Targets
            y_train.append(
                get_knee_point(data_list_train, i) if target=='kp'
                else get_eol_point(eol_def, data_list_train,cycle_list_train,i)
            )
        # Reshaping for input to ML models
        X_train = np.atleast_2d(X_train).reshape(
            len(data_list_train), int(pred_from_cycle/freq)
        )
        y_train = np.atleast_2d(y_train).reshape(-1,1)
        
        # Test data
        X_test = []
        y_test = []
        for i in test_cell_numbers:
            # Prediction inputs
            X_test.append(data_list_o[i][:pred_from_cycle:freq])
            # True output values
            y_test.append(
                get_knee_point(data_list_o, i) if target=='kp'
                else get_eol_point(eol_def, data_list_o, cycle_list_o, i)
            )
        # Reshaping for ML models
        X_test = np.atleast_2d(X_test).reshape(
            len(test_cell_numbers), int(pred_from_cycle/freq)
        )
        y_test = np.atleast_2d(y_test).reshape(-1,1)
        
        # Scaling data
        scalerX1 = StandardScaler().fit(X_train)
        scalery = StandardScaler().fit(y_train)
        X_train = scalerX1.transform(X_train)
        X_test  = scalerX1.transform(X_test)
        y_train = scalery.transform(y_train)
        
        # Error lists
        all_errors = []
        all_percent_err = []
        #Actual
        y_actual = y_test
        
        # Modelling: GPR or CNN
        if model == 'gpr':
            gp = gpr_model(X_train, y_train)
            # Data preprocessing
            pred_input = X_test.reshape(
                len(test_cell_numbers), int(pred_from_cycle/freq)
            )
            # Prediction
            y_mean, sigma = gp.predict(pred_input, return_std=True)
            y_pred = scalery.inverse_transform(y_mean)
        
        if model == 'cnn':
            # Data preprocessing
            complete_inputs = np.empty(
                (X_train.shape[0], X_train.shape[1], 1)
            )
            for i in range(X_train.shape[0]):
              for j in range(X_train.shape[1]):
                  complete_inputs[i][j][0] = X_train[i][j]
            # Reshaping others
            targets = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
            # Test-train split
            complete_inputs, val_inputs, targets, val_targets=train_test_split(
                complete_inputs, targets, test_size=0.15, shuffle=True
            )
            # Callbacks
            cMCP_vl = tf.keras.callbacks.ModelCheckpoint(
                ckpt_val_loss, monitor='val_loss', save_best_only=True,
                mode='min', verbose=False
            )
            callback = [cMCP_vl]
            # Model
            model_cnn = cnn_model(
                complete_inputs, targets, val_inputs, val_targets, callback
            )
            # Saving commands
            model_cnn.save(model_savepath)
            model_cnn = None
            model_cnn = tf.keras.models.load_model(
                model_savepath, compile=False
            )
            model_cnn.load_weights(ckpt_val_loss)
            # Data preprocessing
            input_to_predict = []
            for i in test_cell_numbers:
                temp = data_list_o[i][:pred_from_cycle:freq]
                input_to_predict.append(temp)
            input_to_predict = scalerX1.transform(input_to_predict)
            predict_on = np.empty((
                len(test_cell_numbers), complete_inputs.shape[1], 1
            ))
            for i in range(len(test_cell_numbers)):
                for j in range(complete_inputs.shape[1]):
                    predict_on[i][j][0] = input_to_predict[i][j]
            # Prediction
            y_pred = model_cnn.predict_step(predict_on)
            y_pred = scalery.inverse_transform(y_pred)
            y_pred = y_pred.flatten()

        # ERRORS
        all_errors = abs(y_actual.ravel() - y_pred.ravel())
        all_percent_err = np.divide(all_errors*100, y_actual.ravel())
        # Mean of errors
        avg_percent_error = np.mean(all_percent_err)
        avg_error  = np.mean(all_errors)
        # Storing errors of single element of pred_from_cycle
        avg_percent_error_list.append(avg_percent_error)
        avg_error_list.append(avg_error)

    # Storing errors of pred_from_cycle_list per iteration
    list_pred_errors_percent.append(avg_percent_error_list)
    list_pred_errors_cycles.append(avg_error_list)

#%% Rearranging results
array_pred_errors_percent = np.array(list_pred_errors_percent).T
array_pred_errors_cycles = np.array(list_pred_errors_cycles).T

# Converting to dataframes
df_percent_errors = pd.DataFrame(
    data = array_pred_errors_percent,
    columns = ["Run_{}".format(i+1) for i in range(n_runs)]
)
df_percent_errors['Run_avg'] = df_percent_errors.mean(axis=1)
df_percent_errors['Prediction_from_Cycle'] = pred_from_cycle_list
df_cycle_errors = pd.DataFrame(
    data = array_pred_errors_cycles,
    columns = ["Run_{}".format(i+1) for i in range(n_runs)]
)
df_cycle_errors['Run_avg'] = df_cycle_errors.mean(axis=1)
df_cycle_errors['Prediction_from_Cycle'] = pred_from_cycle_list

# Exporting results
df_percent_errors.to_excel(
    savepath + "Percent_errors_{}_{}".format(target, model) + ".xlsx"
)
df_cycle_errors.to_excel(
    savepath + "Cycle_errors_{}_{}".format(target, model) + ".xlsx"
)

#%% Plotting results
df_percent_errors.plot(x = 'Prediction_from_Cycle', y = 'Run_avg')
plt.xlabel("Input availability in cycles")
plt.ylabel("Error in percent")
plt.show()
plt.savefig(
    savepath + "Average_percent_errors_{}_{}".format(target, model) + ".png"
)
df_cycle_errors.plot(x = 'Prediction_from_Cycle', y = 'Run_avg')
plt.xlabel("Input availability in cycles")
plt.ylabel("Error in cycles")
plt.show()
plt.savefig(
    savepath + "Average_cycle_errors_{}_{}".format(target, model) + ".png"
)

