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

# step 1 - clean the data

# Open data file and remove Id column
print("Creating testing, training and comit dataframes")
input_text_file = pd.read_csv(
    "../data/android/android-commit-messages.csv")
input_training = pd.read_csv(
    "../data/android/training.csv")
input_test = pd.read_csv(
    "../data/android/test.csv")

comment_df = input_text_file.Comment
id_column = input_text_file.Id

# Remove special characters
print("Removing special characters")
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–", "$", "ø", "å"]
comment_df = comment_df.str.replace('|'.join(map(re.escape, spec_chars)), '')
comment_df = comment_df.str.split().str.join(" ")

# Remove Ids
print("Removing IDs")
removed_id = []
for row in comment_df:
    removed_id.append(
        re.sub(r'\w*\d\w*', '', row).strip()
    )

# Stemming, Remove stop words
print("Tokenizing comments")
print("Stemming, removing stop words and characters with a length of 1")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stemmer = PorterStemmer()
stop = stopwords.words('english')

lemma_word = []
place_holder = []
for row in removed_id:
    place_holder = ""
    for word in w_tokenizer.tokenize(row):
        stem = stemmer.stem(word)
        if stem.lower() not in stop and len(stem) > 2:
            place_holder += stem + " "
    lemma_word.append(place_holder)

comment_df = pd.DataFrame(lemma_word, columns=["Comment"])
print("View cleaned data in /part_III/data_cleaned/android/cleaned_data_android.csv")
input_text_file = input_text_file.assign(Comment=comment_df)
input_text_file.to_csv(
    './data_cleaned/android/cleaned_data_android.csv', index=False)


# Bag of Words (1 - Gram)
# Remove words with frequency <= 3
print("Collecting most frequent words with frequency > 3")
comment_df = input_text_file.Comment
wordfreq = {}
to_delete = list()
for row in comment_df:
    for word in w_tokenizer.tokenize(row):
        if word not in wordfreq.keys():
            wordfreq[word] = 1
            to_delete.append(word)
        else:
            wordfreq[word] += 1
            if (wordfreq[word] == 40):
                to_delete.remove(word)

copy_wordfreq = dict(wordfreq)

for (key, value) in copy_wordfreq.items():
    if key in to_delete:
        del wordfreq[key]

print("Creating table for most frequent words given comments")
sentence_vectors = []
count = 1
for sentence in comment_df:
    sentece_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    # print(setence)
    for token in wordfreq:
        if token in sentece_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

word_frequency_android_df = pd.DataFrame(sentence_vectors, columns=wordfreq)

print("View most frequent word table in /part_III/data_cleaned/android/word_frequency_android.csv")
word_frequency_android_df = word_frequency_android_df.assign(
    Id=input_text_file.Id)
word_frequency_android_df.to_csv(
    './data_cleaned/android/word_frequency_android.csv', index=False)

print("View final training data with bag of words in /part_III/data_cleaned/android/training_data_final.csv")
training_data_final_df = pd.merge(left=input_training, right=word_frequency_android_df,
                                  how='left', left_on='Id', right_on='Id')
training_data_final_df.to_csv(
    './data_cleaned/android/training_data_final.csv', index=False)

print("View final testing data with bag of words in part_III/data_cleaned/android/testing_data_final.csv")
testing_data_final_df = pd.merge(left=input_test, right=word_frequency_android_df,
                                 how='left', left_on='Id', right_on='Id')
testing_data_final_df.to_csv(
    './data_cleaned/android/testing_data_final.csv', index=False)
