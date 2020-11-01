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
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# import adtrees as adt

input_file_training = "../data/openstack/training.csv"
input_file_test = "../data/openstack/test.csv"
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
                    DecisionTreeClassifier(),
                    RandomForestClassifier()]

for classifier in classifier_array:
    test_pred = classifier.fit(train_data, train_target).predict(test_data)
    print(classifier, '\n', classification_report(
        test_target, test_pred, labels=[0, 1]))
    if classifier == classifier_array[0]:
        max_iter = [4, 5, 100, 120, 1500]
        random_parameter_values = {'max_iter': max_iter}
        rf = LogisticRegression()
        rf_random = GridSearchCV(estimator=rf, param_grid=random_parameter_values,
                                 cv=3, verbose=2, n_jobs=-1, scoring="f1_weighted")
        # Fit the random search model
        test_pred = rf_random.fit(train_data, train_target).predict(test_data)
        score_csv = rf_random.cv_results_
        score_csv_df = pd.DataFrame(score_csv)
        score_csv_df.to_csv(
            './score/openstack/logistic_regression_openstack.csv', index=False)
        print('\n Best Estimator: ', rf_random.best_estimator_)
        print('\n LogisticRegression() - With Best Estimator Parameter Values')
        rf_best = rf_random.best_estimator_.fit(
            train_data, train_target).predict(test_data)
        print(classification_report(test_target, test_pred, labels=[0, 1]))
    elif classifier == classifier_array[1]:
        max_depth = [3, 70, 200, 500]
        max_depth.append(None)
        max_leaf_nodes = [8, 4000, 600, 1200]
        max_leaf_nodes.append(None)
        min_samples_split = [1.0, 2, 10, 20, 100]
        random_parameter_values = {'max_depth': max_depth,
                                   'max_leaf_nodes': max_leaf_nodes,
                                   'min_samples_split': min_samples_split}
        rf = DecisionTreeClassifier()
        rf_random = GridSearchCV(estimator=rf, param_grid=random_parameter_values,
                                 cv=3, verbose=2, n_jobs=-1, scoring="f1_weighted")
        # Fit the random search model
        test_pred = rf_random.fit(train_data, train_target).predict(test_data)
        score_csv = rf_random.cv_results_
        score_csv_df = pd.DataFrame(score_csv)
        score_csv_df.to_csv(
            './score/openstack/decision_tree_openstack.csv', index=False)
        print('\n Best Estimator: ', rf_random.best_estimator_)
        print('\n DecisionTreeClassifier() - With Best Estimator Parameter Values')
        rf_best = rf_random.best_estimator_.fit(
            train_data, train_target).predict(test_data)
        print(classification_report(test_target, test_pred, labels=[0, 1]))
    elif classifier == classifier_array[2]:  # RandomForestClassifier()
        max_depth = [10, 2000, 400, 2]
        max_depth.append(None)
        bootstrap = [True, False]
        max_leaf_nodes = [8, 4000, 600, 1200]
        max_leaf_nodes.append(None)
        random_parameter_values = {
            'max_depth': max_depth,
            'bootstrap': bootstrap,
            'max_leaf_nodes': max_leaf_nodes}
        rf = RandomForestClassifier()
        rf_random = GridSearchCV(estimator=rf, param_grid=random_parameter_values,
                                 cv=3, verbose=2, n_jobs=-1, scoring="f1_weighted")
        # Fit the random search model
        test_pred = rf_random.fit(train_data, train_target).predict(test_data)
        score_csv = rf_random.cv_results_
        score_csv_df = pd.DataFrame(score_csv)
        score_csv_df.to_csv(
            './score/openstack/random_forest_openstack.csv', index=False)
        print('\n Best Estimator: ', rf_random.best_estimator_)
        print('\n RandomForestClassifier() - With Best Estimator Parameter Values')
        rf_best = rf_random.best_estimator_.fit(
            train_data, train_target).predict(test_data)
        print(classification_report(test_target, test_pred, labels=[0, 1]))
