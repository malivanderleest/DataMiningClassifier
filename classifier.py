# Libraries
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pprint

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
    validation_data = validation_data.sample(frac=1)


    # Divide remaining data into 10-fold subsets
    kfold_class = training_data['J']

    validation_y = validation_data['J']
    validation_X = validation_data
    del validation_X['J']

    ################################ HYPERPARAMETERS #################################
    # k-nearest neighbors
    k_params = [1,2,3,4,5,6,7,8,9,10]
    k_params_accuracy = [0,0,0,0,0,0,0,0,0,0]

    # Decision tree
    max_depth_params = [1,2,3,4,5,6,7,8,9,10]
    max_depth_params_accuracy = [0,0,0,0,0,0,0,0,0,0]
    
    # Random Forest
    trees_params = [25,50,100,125,150,175,200,225,250,275]
    #trees_params = [1,1,1,1,1,1,1,1,1,1,1]
    trees_params_accuracy = [0,0,0,0,0,0,0,0,0,0]

    # SVM using the polynomial kernel
    d_params = [1,2,3,4,5,6,7,8,9,10]
    d_params_accuracy = [0,0,0,0,0,0,0,0,0,0]
    
    # SVM using the RBF kernel
    gamma_params = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    gamma_params_accuracy = [0,0,0,0,0,0,0,0,0,0]
    
    # Deep neural network with sigmoid activation
    neurons_params = [1,2,3,4,5,6,7,8,9,10]
    neurons_params_accuracy = [0,0,0,0,0,0,0,0,0,0]
    
    # Deep neural network with ReLU activation
    relu_neurons_params = [1,2,3,4,5,6,7,8,9,10]
    relu_neurons_params_accuracy = [0,0,0,0,0,0,0,0,0,0]

    index = 0
    skf = StratifiedKFold(n_splits=10)

    del training_data['J']
    #print(training_data)

    while index < 10:

        k_total = 0
        max_depth_total = 0
        trees_total = 0
        d_total = 0
        gamma_total = 0
        neurons_total = 0
        relu_neurons_total = 0

        knn_max = 0
        tree_max = 0
        forest_max = 0
        poly_max = 0
        rbf_max = 0
        sigmoid_max = 0
        relu_max = 0
        
        knn_score = 0
        tree_score = 0
        forest_score = 0
        poly_score = 0
        rbf_score = 0
        sigmoid_score = 0
        relu_score = 0

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
            if knn.score(X_test, y_test) > knn_score:
                knn_max = k_param
                knn_score = knn.score(validation_X, validation_y)

            # Decision tree classifier with different max depths
            depth = max_depth_params[index]
            dectree = tree.DecisionTreeClassifier(max_depth = depth)
            dectree.fit(X_train, y_train)
            # Decision tree testing
            max_depth_total = max_depth_total + dectree.score(X_test, y_test)
            if dectree.score(X_test, y_test) > tree_score:
                tree_max = depth
                tree_score = dectree.score(validation_X, validation_y)

            # Random forest classifier with different numbers of trees
            num_trees = trees_params[index]
            forest = RandomForestClassifier(n_estimators = num_trees)
            forest.fit(X_train, y_train)
            # Random forest testing
            trees_total = trees_total + forest.score(X_test,y_test)
            if forest.score(X_test, y_test) > forest_score:
                forest_max = num_trees
                forest_score = forest.score(validation_X, validation_y)

            # SVM polynomial kernel classifier
            d = d_params[index]
            poly_svm = SVC(kernel = 'poly', degree = d)
            poly_svm.fit(X_train, y_train)
            # SVM polynomial kernel testing
            d_total = d_total + poly_svm.score(X_test,y_test)
            if poly_svm.score(X_test, y_test) > poly_score:
                poly_max = d
                poly_score = poly_svm.score(validation_X, validation_y)

            # SVM RBF kernel classifier
            gamma_val = gamma_params[index]
            rbf_svm = SVC(kernel = 'rbf', gamma = gamma_val)
            rbf_svm.fit(X_train, y_train)
            # SVM RBF testing
            gamma_total = gamma_total + rbf_svm.score(X_test,y_test)
            if rbf_svm.score(X_test, y_test) > rbf_score:
                rbf_max = gamma_val
                rbf_score = rbf_svm.score(validation_X, validation_y)

            # Sigmoid DNN classifier
            neurons = neurons_params[index]
            sigmoid_dnn = MLPClassifier(hidden_layer_sizes = neurons, 
                                        activation = 'logistic', 
                                        max_iter = 2000)
            sigmoid_dnn.fit(X_train, y_train)
            # Sigmoid DNN testing
            neurons_total = neurons_total + sigmoid_dnn.score(X_test,y_test)
            if sigmoid_dnn.score(X_test, y_test) > sigmoid_score:
                sigmoid_max = neurons
                sigmoid_score = sigmoid_dnn.score(validation_X, validation_y)

            # ReLu DNN classifier
            relu_neurons = relu_neurons_params[index]
            relu_dnn = MLPClassifier(hidden_layer_sizes = neurons, max_iter = 2000)
            relu_dnn.fit(X_train, y_train)
            # ReLu DNN testing
            relu_neurons_total = relu_neurons_total + relu_dnn.score(X_test,y_test)
            if relu_dnn.score(X_test, y_test) > relu_score:
                relu_max = relu_neurons
                relu_score = relu_dnn.score(validation_X, validation_y)

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

        # Average accuracy for RBF gamma values
        avg_gamma_accuracy = gamma_total/10
        gamma_params_accuracy[index] = avg_gamma_accuracy

        # Average accuracy for sigmoid DNN number of neurons
        avg_neurons_accuracy = neurons_total/10
        neurons_params_accuracy[index] = avg_neurons_accuracy

        # Average accuracy for sigmoid DNN number of neurons
        avg_relu_neurons_accuracy = relu_neurons_total/10
        relu_neurons_params_accuracy[index] = avg_relu_neurons_accuracy

        #print("DEPTH = ", depth)
        #r = tree.export_text(dectree, feature_names=['A','B','C','D','E','F','G','H','I'])
        #print(r)

        index = index + 1

    print("************* knn ***************")
    k_params_map = list(zip(k_params, k_params_accuracy))
    pprint.pprint(k_params_map)

    print("\n************* decision tree ***************")
    max_depth_params_map = list(zip(max_depth_params, max_depth_params_accuracy))
    pprint.pprint(max_depth_params_map)

    print("\n************* random forest ***************")
    trees_params_map = list(zip(trees_params, trees_params_accuracy))
    pprint.pprint(trees_params_map)

    print("\n************* svm poly kernel ***************")
    d_params_map = list(zip(d_params, d_params_accuracy))
    pprint.pprint(d_params_map)

    print("\n************* svm rbf kernel ***************")
    gamma_params_map = list(zip(gamma_params, gamma_params_accuracy))
    pprint.pprint(gamma_params_map)

    print("\n************* dnn sigmoid ***************")
    neurons_params_map = list(zip(neurons_params, neurons_params_accuracy))
    pprint.pprint(neurons_params_map)

    print("\n************* dnn relu ***************")
    relu_neurons_params_map = list(zip(relu_neurons_params, relu_neurons_params_accuracy))
    pprint.pprint(relu_neurons_params_map)

   
    ############################################################################

    # Hyperparameter accuracy plots
    if plot:
        plt.scatter(k_params,k_params_accuracy)
        plt.plot(list(k_params),list(k_params_accuracy))
        plt.title('Accuracy of KNN classifier for different k values')
        plt.xlabel('k (number of nearest neighbors)')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('knn.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        
        plt.scatter(max_depth_params,max_depth_params_accuracy)
        plt.plot(list(max_depth_params),list(max_depth_params_accuracy))
        plt.title('Accuracy of decision tree classifier for different max depth values')
        plt.xlabel('max depth of tree')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('dectree.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure

        plt.scatter(trees_params, trees_params_accuracy)
        plt.plot(list(trees_params),list(trees_params_accuracy))
        plt.title('Accuracy of random forest classifier for different quantities of trees')
        plt.xlabel('number of trees in forest')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('randomforest.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure

        plt.scatter(d_params,d_params_accuracy)
        plt.plot(list(d_params),list(d_params_accuracy))
        plt.title('Accuracy of SVM polynomial kernel classifier for different d values')
        plt.xlabel('d (degree of polynomial kernel)')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('svmpoly.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure

        plt.scatter(gamma_params,gamma_params_accuracy)
        plt.plot(list(gamma_params),list(gamma_params_accuracy))
        plt.title('Accuracy of SVM RBF classifier for different gamma values')
        plt.xlabel('gamma')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('svmrbf.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure

        plt.scatter(neurons_params,neurons_params_accuracy)
        plt.plot(list(neurons_params),list(neurons_params_accuracy))
        plt.title('Accuracy of DNN classifier with Sigmoid activation function for different quantities of neurons in hidden layers')
        plt.xlabel('number of neurons in hidden layers')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('dnnsigmoid.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure

        plt.scatter(relu_neurons_params,relu_neurons_params_accuracy)
        plt.plot(list(relu_neurons_params),list(relu_neurons_params_accuracy))
        plt.title('Accuracy of DNN classifier with ReLu activation function for different quantities of neurons in hidden layers')
        plt.xlabel('number of neurons in hidden layers')
        plt.ylabel('Accuracy of classifier')
        plt.ylim([.8, 1])
        plt.show()
        plt.savefig('dnnrelu.png')
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure

        print("********* VALIDATION SCORES *****************")
        print(knn_score)
        print(tree_score)
        print(forest_score)
        print(poly_score)
        print(rbf_score)
        print(sigmoid_score)
        print(relu_dnn)
