import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# print(test_target)
classifier_array = [LogisticRegression(),
                    KNeighborsClassifier(),
                    GaussianNB(priors=None),
                    RandomForestClassifier()]

for classifier in classifier_array:
    test_pred = classifier.fit(train_data, train_target).predict(test_data)
    print(classifier, '\n', classification_report(
        test_target, test_pred, labels=[0, 1]))
