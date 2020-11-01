import numpy as np
import csv as csv
import pandas as pd
import re
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import heapq

df1 = pd.DataFrame({'id': [1, 2, 3, 4], 'lkey': ['foo', 'bar', 'baz', 'foo'],
                    'valuha': [1, 2, 3, 5]})
df2 = pd.DataFrame({'id': [1, 11, 2, 3, 4], 'rkey': ['hi', 'foo', 'bar', 'baz', 'foo'],
                    'vadfdlue': [3, 5, 6, 7, 8]})
# df1 = df1.merge(df2, left_on='lkey', right_on='rkey',
#                 suffixes=('_left', '_right'))
# print(df1)

dmerged_left = pd.merge(left=df1, right=df2,
                        how='left', left_on='id', right_on='id')
print(dmerged_left)
