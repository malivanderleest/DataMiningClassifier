# Libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.model_selection import StratifiedKFold



#-------------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------------
# Run main program
if __name__ == "__main__":
    # Parse command line input
    plot = False
    if '-plot' in sys.argv:
        plot = True

    ################################ DATA SETS #################################
    # .csv file containing multivariate breast cancer data
    data = pd.read_csv('cancer_data.csv')

    # Drop all samples with unknown values
    data = data.loc[data['F'] != '?']

    # Separate benign and malignant tumor data
    b_data = data.loc[data['J'] == 2]
    m_data = data.loc[data['J'] == 4]

    # Allocate 10% to a validation set and maintain ratios
    b_validation = b_data.sample(frac = 0.1)
    data = data.drop(b_validation.index)
    m_validation = m_data.sample(frac = 0.1)
    data = data.drop(m_validation.index)
    validation_data = b_validation.append(m_validation, shuffle = True)

    # Divide remaining data into 10-fold subsets
    kfold_class = data['J']

    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(data, kfold_class):
        print("\n**************** SUBSET ****************")
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = kfold_class.iloc[train_index], kfold_class.iloc[test_index]

        # k-nearest neighbors
        k_param_accuracy = {}

        # Decision tree
        first_split_param_accuracy = {}
        
        # Random Forest
        first_split_param_accuracy = {}

        # SVM using the polynomial kernel
        d_param_accuracy = {}
        
        # SVM using the RBF kernel
        gamma_param_accuracy = {}
        
        # Deep neural network with sigmoid activation
        z_param_accuracy = {}
        
        # Deep neural network with ReLU activation
        z_param_accuracy = {}
    
   
    ############################################################################



    # Plot univariate regression models
    
    # if plot:
    #     plt.scatter(training['training_x1'], training['training_y'])
    #     plt.title('Component 1 Trained Univariate Model')
    #     plt.xlabel('Cement (kg in a m^3 mixture)')
    #     plt.ylabel('Concrete Compressive Strength (MPa)')
    #     plt.plot(training['training_x1'], updated_m_params[0]*training['training_x1']+updated_b_params[0], color = "black")
    #     plt.show()
