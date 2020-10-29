import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# import adtrees as adt

input_file_training = "../data/android/training.csv"
input_file_test = "../data/android/test.csv"
# load the training data as a matrix
dataset = pd.read_csv(input_file_training, header=0)
# separate the data from the target attributes
train_data = dataset.drop('isReq', axis=1)
# remove unnecessary features
train_data = train_data.drop('Id', axis=1)
# the lables of training data. `label` is the title of the  last column in your CSV files
train_target = dataset.isReq
# load the testing data
dataset2 = pd.read_csv(input_file_test, header=0)
# separate the data from the target attributes
test_data = dataset2.drop('isReq', axis=1)
# remove unnecessary features
test_data = test_data.drop('Id', axis=1)
# the lables of test data
test_target = dataset2.isReq

# print('testarget', test_target)

classifier_array = [LogisticRegression(),
                    KNeighborsClassifier(),
                    RandomForestClassifier()]

for classifier in classifier_array:
    test_pred = classifier.fit(train_data, train_target).predict(test_data)
    print(classifier, '\n', classification_report(
        test_target, test_pred, labels=[0, 1]))
    if classifier == classifier_array[0]:
        max_iter = [4, 5, 100, 120, 150]
        random_parameter_values = {'max_iter': max_iter}
        rf = LogisticRegression()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_parameter_values,
                                       n_iter=2, cv=2, verbose=2, random_state=42, n_jobs=-1, scoring="f1_weighted")
        # Fit the random search model
        test_pred = rf_random.fit(train_data, train_target).predict(test_data)
        print('\n Best Estimator: ', rf_random.best_estimator_)
        print('\n LogisticRegression() - With Best Estimator Parameter Values')
        # pprint(rf_random.best_score_) MSU
        rf_best = rf_random.best_estimator_.fit(
            train_data, train_target).predict(test_data)
        print(classification_report(test_target, test_pred, labels=[0, 1]))
    elif classifier == classifier_array[1]:
        n_neighbors = [3, 4, 5, 10, 7]
        leaf_size = [20, 25, 30, 35, 40]
        random_parameter_values = {'n_neighbors': n_neighbors,
                                   'leaf_size': leaf_size}
        rf = KNeighborsClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_parameter_values,
                                       n_iter=4, cv=2, verbose=2, random_state=42, n_jobs=-1, scoring="f1_weighted")
        # Fit the random search model
        test_pred = rf_random.fit(train_data, train_target).predict(test_data)
        # print('\n Best Estimator: ', rf_random.best_estimator_)
        print('\n KNeighborsClassifier() - With Best Estimator Parameter Values')
        # pprint(rf_random.best_score_) MSU
        rf_best = rf_random.best_estimator_.fit(
            train_data, train_target).predict(test_data)
        print(classification_report(test_target, test_pred, labels=[0, 1]))
    elif classifier == classifier_array[2]:  # RandomForestClassifier()
        n_estimators = [int(x)
                        for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        bootstrap = [True, False]
        max_leaf_nodes = [2, 3, 4, 6, 10]
        max_leaf_nodes.append(None)
        random_parameter_values = {'n_estimators': n_estimators,
                                   'max_features': max_features,
                                   'max_depth': max_depth,
                                   'bootstrap': bootstrap,
                                   'max_leaf_nodes': max_leaf_nodes}
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_parameter_values,
                                       n_iter=4, cv=2, verbose=2, random_state=42, n_jobs=-1, scoring="f1_weighted")
        # Fit the random search model
        test_pred = rf_random.fit(train_data, train_target).predict(test_data)
        # print('\n Best Estimator: ', rf_random.best_estimator_)
        print('\n RandomForestClassifier() - With Best Estimator Parameter Values')
        # pprint(rf_random.best_score_) MSU
        rf_best = rf_random.best_estimator_.fit(
            train_data, train_target).predict(test_data)
        print(classification_report(test_target, test_pred, labels=[0, 1]))
