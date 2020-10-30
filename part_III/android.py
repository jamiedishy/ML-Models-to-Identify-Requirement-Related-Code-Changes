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
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pdpipe as pdp

# ps = PorterStemmer()
# import adtrees as adt

# step 1 - clean the data
input_text_file = pd.read_csv(
    "../data/android/android-commit-messages.txt")
input_text_file.to_csv(
    './data_cleaned/android/android-commit-messages.csv')
input_commit_file = pd.read_csv(
    './data_cleaned/android/android-commit-messages.csv')

input_commit_file.drop('Id', inplace=True, axis=1)
input_commit_file.to_csv(
    './data_cleaned/android/android-commit-messages.csv')
input_commit_file.drop('Unnamed: 0', inplace=True, axis=1)
input_commit_file.to_csv(
    './data_cleaned/android/android-commit-messages.csv')

comment = input_commit_file.Comment

# Remove special characters
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–"]
for char in spec_chars:
    comment = comment.str.replace(char, ' ')
comment = comment.str.split().str.join(" ")

# Stematize
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
# stemmer = PorterStemmer()
# print(stemmer.stem('frightening'))
# print(lemmatizer.lemmatize('frightening'))


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


comment = comment.apply(lemmatize_text)


# Remove stop words
stop = stopwords.words('english')
comment = comment.apply(lambda x: [item for item in x if item not in stop])

# Rejoin tokenized column
# for row in comment:
#     TreebankWordDetokenizer().detokenize(row)
# comment = comment.apply(lambda x: [for row in comment: ])

comment.to_csv('./data_cleaned/android/removed_stop_words.csv')
