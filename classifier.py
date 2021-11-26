# Libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk


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

    print(data.shape)
    # Preprocess the data
    data = data.loc[data['F'] != '?']
    print(data.shape)

    
    # Separate training data into univariate lists
    training = {
        #'training_x1' : training_data['cement'],
    }
    ############################################################################



    # Plot univariate regression models
    
    # if plot:
    #     plt.scatter(training['training_x1'], training['training_y'])
    #     plt.title('Component 1 Trained Univariate Model')
    #     plt.xlabel('Cement (kg in a m^3 mixture)')
    #     plt.ylabel('Concrete Compressive Strength (MPa)')
    #     plt.plot(training['training_x1'], updated_m_params[0]*training['training_x1']+updated_b_params[0], color = "black")
    #     plt.show()
