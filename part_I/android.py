import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

input_file_training = pd.read_csv("../data/android/training.csv", header=0)
input_file_test = "../data/android/test.csv"


# target_count = input_file_training.isReq.value_counts()
# print('class 0: ', target_count[0])
# print('class 1: ', target_count[1])
# print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

countclass0, countclass1 = input_file_training.isReq.value_counts()

class1 = input_file_training[input_file_training['isReq'] == 1]
class0 = input_file_training[input_file_training['isReq'] == 0]

class1_over = class1.sample(countclass0, replace=True)
testover = pd.concat([class0, class1_over], axis=0)

print('Random over-sampling: ')
print(testover.isReq.value_counts())

# testover.isReq.value_counts().plot(kind='bar', title='Count (target)')


# # load the training data as a matrix
# dataset = pd.read_csv(input_file_training, header=0)

# # separate the data from the target attributes
train_data = testover.drop('isReq', axis=1)


# # remove unnecessary features
train_data = train_data.drop('Id', axis=1)

# # the lables of training data. `label` is the title of the  last column in your CSV files
train_target = testover.isReq

# # load the testing data
dataset2 = pd.read_csv(input_file_test, header=0)

# # separate the data from the target attributes
test_data = dataset2.drop('isReq', axis=1)

# # remove unnecessary features
test_data = test_data.drop('Id', axis=1)

# # the lables of test data
test_target = dataset2.isReq

# # print(test_target)

gnb = GaussianNB()
# #decision_t = DecisionTreeClassifier(random_state=0, max_depth=2)
test_pred = gnb.fit(train_data, train_target).predict(test_data)

print(classification_report(test_target, test_pred, labels=[0, 1]))
