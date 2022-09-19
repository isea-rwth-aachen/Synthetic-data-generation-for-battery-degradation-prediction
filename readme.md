# Synthetic data generation for battery degradation prediction

# Introduction
The data and code in this repository associated with the research work at ISEA, RWTH Aachen University in synthetic data generation for battery degradation prediction.

# Datasets

# Modeling Codes

# General Steps

1. Input capacity/time-series data file should be of the format as shown in the
   included open source NASA Battery Dataset titled 'NASA_dataset.xlsx'.

2. For generating multiple synthetic capacity/time-series data per cell,
   please create a new array with a different name. For example:
       synthetic_data_list_1 = random_synthetic_data(...)
       synthetic_data_list_2 = random_synthetic_data(...)

3. For generating synthetic curves of all cells having the exact same
   synthetic curve generation parameter(s), please enter the same value
   of the upper and lower bounds for the parameter(s).

4. For generating synthetic curves only for some specific cells within the
   dataset, please enter the indices of the cell data within the
   'selected_cells' array. For example:
       selected_cells = np.array([1, 3]) # Cell numbers 1 and 3 selected

# Contact
Weihan Li weihan.li@isea.rwth-aachen.de
Harshvardhan Samsukha harshvardhan.samsukha@rwth-aachen.de 
