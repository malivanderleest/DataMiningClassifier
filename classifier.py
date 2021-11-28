# Libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

    # Shuffle values
    training_data = data.sample(frac=1)
    del training_data['ID']

    # Separate benign and malignant tumor data
    b_data = training_data.loc[training_data['J'] == 2]
    m_data = training_data.loc[training_data['J'] == 4]

    # Allocate 10% to a validation set and maintain ratios
    b_validation = b_data.sample(frac = 0.1)
    training_data = training_data.drop(b_validation.index)
    m_validation = m_data.sample(frac = 0.1)
    training_data = training_data.drop(m_validation.index)
    validation_data = b_validation.append(m_validation)

    # Divide remaining data into 10-fold subsets
    kfold_class = training_data['J']

    ################################ HYPERPARAMETERS #################################
    # k-nearest neighbors
    k_params = [1,2,4,8,16,32,64,128,256,512]
    k_params_accuracy = [0,0,0,0,0,0,0,0,0,0]

    # Decision tree
    max_depth_params = [1,2,3,4,5,6,7,8,9,10]
    max_depth_params_accuracy = [0,0,0,0,0,0,0,0,0,0]
    
    # Random Forest
    trees_params = [25,50,100,125,150,175,200,225,250,275]
    trees_params_accuracy = [0,0,0,0,0,0,0,0,0,0]

    # SVM using the polynomial kernel
    d_params = [1,2,3,4,5,6,7,8,9,10]
    d_params_accuracy = [0,0,0,0,0,0,0,0,0,0]
    
    # SVM using the RBF kernel
    gamma_params = []
    gamma_params_accuracy = []
    
    # Deep neural network with sigmoid activation
    z_params = []
    z_params_accuracy = []
    
    # Deep neural network with ReLU activation
    relu_z_params = []
    relu_z_params_accuracy = []

    index = 0
    skf = StratifiedKFold(n_splits=10)

    del training_data['J']
    print(training_data)

    while index < 10:

        k_total = 0
        max_depth_total = 0
        trees_total = 0
        d_total = 0

    ################################ TRAINING #################################
        for train_index, test_index in skf.split(training_data, kfold_class):
            #print("\n**************** SUBSET ****************")
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = training_data.iloc[train_index], training_data.iloc[test_index]
            y_train, y_test = kfold_class.iloc[train_index], kfold_class.iloc[test_index]

            # KNN classifier training using Euclidean distance
            k_param = k_params[index]
            knn = KNeighborsClassifier(n_neighbors = k_param)
            knn.fit(X_train,y_train)
            #KNN classifier testing
            k_total = k_total + knn.score(X_test, y_test)

            # Decision tree classifier with different max depths
            depth = max_depth_params[index]
            dectree = tree.DecisionTreeClassifier(max_depth = depth)
            dectree = dectree.fit(X_train, y_train)
            # Decision tree testing
            max_depth_total = max_depth_total + dectree.score(X_test, y_test)

            # Random forest classifier with different numbers of trees
            num_trees = trees_params[index]
            forest = RandomForestClassifier(n_estimators = num_trees)
            forest.fit(X_train, y_train)
            # Random forest testing
            trees_total = trees_total + forest.score(X_test,y_test)

            # SVM polynomial kernel classifier
            d = d_params[index]
            poly_svm = SVC(kernel = 'poly', degree = d)
            poly_svm.fit(X_train, y_train)
            # SVM polynomial kernel testing
            d_total = d_total + poly_svm.score(X_test,y_test)

    ################################ AVG ACCURACY #################################
        # Average accuracy for k
        avg_k_accuracy = k_total/10
        k_params_accuracy[index] = avg_k_accuracy

        # Average accuracy for decision tree depth
        avg_max_depth_accuracy = max_depth_total/10
        max_depth_params_accuracy[index] = avg_max_depth_accuracy

        # Average accuracy for forest number of trees
        avg_trees_accuracy = trees_total/10
        trees_params_accuracy[index] = avg_trees_accuracy

        # Average accuracy for polynomial kernel degrees
        avg_d_accuracy = d_total/10
        d_params_accuracy[index] = avg_d_accuracy

        #print("DEPTH = ", depth)
        #r = tree.export_text(dectree, feature_names=['A','B','C','D','E','F','G','H','I'])
        #print(r)

        index = index + 1

    print("************* knn ***************")
    print(k_params_accuracy)

    print("\n************* decision tree ***************")
    print(max_depth_params_accuracy)

    print("\n************* random forest ***************")
    print(trees_params_accuracy)

    print("\n************* svm poly kernel ***************")
    print(d_params_accuracy)

   
    ############################################################################



    # Plot univariate regression models
    
    # if plot:
    #     plt.scatter(training['training_x1'], training['training_y'])
    #     plt.title('Component 1 Trained Univariate Model')
    #     plt.xlabel('Cement (kg in a m^3 mixture)')
    #     plt.ylabel('Concrete Compressive Strength (MPa)')
    #     plt.plot(training['training_x1'], updated_m_params[0]*training['training_x1']+updated_b_params[0], color = "black")
    #     plt.show()
