# Synthetic data generation for battery degradation prediction

# Introduction
The data and code in this repository associated with the research work at ISEA, RWTH Aachen University in synthetic data generation for battery degradation prediction.

# Datasets
The following four datasets have been used to generate and validate the results in the associated paper.
	1. RWTH Dataset, Source: D. Beck, P. Dechent, M. Junker, D. U. Sauer, and M. Dubarry, “Inhomogeneities and Cell-to-Cell Variations in Lithium-Ion Batteries, a Review,” Energies, vol. 14, no. 11, p. 3276, 2021, doi: 10.3390/en14113276.
	2. Stanford Dataset, Source: P. M. Attia et al., "Closed-loop optimization of fast-charging protocols for batteries with machine learning," Nature, vol. 578, no. 7795, pp. 397–402, 2020, doi: 10.1016/j.jpowsour.2004.12.038.
	3. NASA Battery Data Set, Source: B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA, B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository.
	4. Oxford Battery Degradation Dataset 1, Source: C. Birkl, "Oxford Battery Degradation Dataset 1," 2017.
	
The dataset files have been preprocessed and stored in the MS Excel file format, which can be accessed within the 'data' folder. The Stanford Dataset was further organized within the Excel file 'Stanford_Dataset_test_wise.xlsx' into 9 separate sheets, each having cell data corresponding to the 9 different test conditions. This dataset's validation was performed through the modelling script 'prediction_stanford_dataset.py', whereas the rest of the datasets were validated using the 'prediction_single_profile.py' script.

# Modeling Codes
All executable python scripts are present within the main directory of the git repository. The description of the individual scripts are as follows:
	1. 'plot_synthetic_data.py': This script plots the synthetic data and real data. The utility of the script is to visualize the output synthetic data curves of the synthetic data generation parameters, which in turns helps in fine-tuning the parameters to generate the desired set of synthetic data. The tuned parameters can then be used within the other modelling scripts.
	2. 'prediction_single_profile.py': This script generates the synthetic data as desired, validates the prediction model results and stores result data in the 'results' folder. Some of the parameters to be tuned are:
		i.   n_training_samples: Specifies the number of real cell data out of the total available cell data in the Excel file that would be used for training the prediction models.
		ii.  n_syn_curves: Specifies the number of synthetic cell data to be generated from the chosen real cell data.
		iii. n_runs: Specifies the number of runs / iterations / loops. Each loop executes the selection of randomly selected 'n_training_samples' number of real data, creates 'n_syn_curves' number of synthetic data, trains and predicts the output of the test cell data specified in the script.
		iv.  model: Selects either the Convolutional Neural Network (string: 'cnn') or Gaussian Process Regression (string: 'gpr') models for training and prediction.
		v.   target: Selects either the End-of-life (EOL) (string: 'eol') or Knee-point (string: 'kp') as the feature of the test cells to be predicted.
		vi.  eol_def: Defines the EOL of cell as a ratio of initial capacity.
		vii. Synthetic data parameters: Defined as a list containing the lower and upper boundaries respectively as list elements.
		viii.test_cell_numbers: Select the test cell indices. The indices are based on the order of the cells as they appear within the Excel file from left to right, the index of the first cell data being 0.
	3. 'prediction_stanford_dataset.py': Due to a different file layout within the Stanford_Dataset_test_wise.xlsx file, some rearrangements were necessary to access cell-data belonging to different test groups. The parameters apart from the aforementioned ones to be tuned are:
		i.   test_column: Selects the same column in each Excel sheet within the Excel file that would be used as validation data for prediction models.
		ii.  n_training_samples_per_batch: Specifies the number of real cell data out of the total available cell data per test condition in a single Excel sheet within the Excel file that would be used for training the prediction models.

Other folders present within the main directory are:
	1. data: Contains the data files for the respective datasets used within the paper.
	2. functions: Contains the functions and models used within the main scripts. The files within are:
		i.  'data_preprocessing.py': Preprocesses data for input to prediction models.
		ii. 'kneepoint_eol_detection.py': Functions that extract the EOL and knee-points of cells from cell data.
		iii.'models.py': Contains model code for the CNN and GPR models. The changes to the parameters of these models can be made within this file to optimize model performance.
		iv. 'plotting.py': Contains functions for visual representation of results.
		v.  'synthetic_data_generator.py': Contains function that generates synthetic data from real data.
	3. results: Stores model prediction results in the form of plots and Excel files.
	

# Miscellaneous Steps

   1. Input capacity/time-series data file should be of the format as shown in the 'data' folder.

   2. For generating synthetic curves only for some specific cells within the dataset, please enter the indices of the cell data within the 'selected_cells' array. For example:
   selected_cells = np.array([1, 3]) # Cells 1 and 3 based on the index of occurrence selected.

# Contact
Weihan Li weihan.li@isea.rwth-aachen.de
Harshvardhan Samsukha harshvardhan.samsukha@rwth-aachen.de 
