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

# step 1 - clean the data

# Open data file and remove Id column
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
              "`", "{", "|", "}", "~", "–", "nChange", "init", "Sig", " n n ",
              " off by ", "\n"]
comment = comment.str.replace('|'.join(map(re.escape, spec_chars)), '')
comment = comment.str.split().str.join(" ")

# Remove Ids
removed_id = []
for row in comment:
    removed_id.append(
        re.sub(r'\w*\d\w*', '', row).strip()
    )

# Stemming
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stemmer = PorterStemmer()

lemma_word = []
place_holder = []
for row in removed_id:
    place_holder = ""
    for word in w_tokenizer.tokenize(row):
        place_holder += stemmer.stem(word) + " "
    lemma_word.append(place_holder)

df = pd.DataFrame(lemma_word, columns=["Comment"])
comment = df

# Remove stop words
stop = stopwords.words('english')
comment = comment.apply(lambda x: [item for item in x if item not in stop])

comment.to_csv('./data_cleaned/android/removed_stop_words.csv')
