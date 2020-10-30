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

# step 1 - clean the data

# Open data file and remove Id column
input_text_file = pd.read_csv(
    "../data/android/android-commit-messages.csv")

comment = input_text_file.Comment
id_column = input_text_file.Id

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

input_text_file = input_text_file.assign(Comment=comment)
input_text_file.to_csv(
    './data_cleaned/android/cleaned_data_android.csv', index=False)


# Bag of Words (1 - Gram)
# Tokenize Comment section
comment = input_text_file.Comment
wordfreq = {}
to_delete = list()
for row in comment:
    for word in w_tokenizer.tokenize(row):
        if word not in wordfreq.keys():
            wordfreq[word] = 1
            to_delete.append(word)
        else:
            wordfreq[word] += 1
            if (wordfreq[word] == 4):
                to_delete.remove(word)

copy_wordfreq = dict(wordfreq)

for (key, value) in copy_wordfreq.items():
    if key in to_delete:
        del wordfreq[key]
