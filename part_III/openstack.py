import numpy as np
import csv as csv
import pandas as pd
import re
import nltk
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer


# Open data file and remove Id column
print("Creating testing, training and commit dataframes")
input_text_file = pd.read_csv(
    "../data/openstack/openstack-commit-messages.csv")
input_training = pd.read_csv(
    "../data/openstack/training.csv")
input_test = pd.read_csv(
    "../data/openstack/test.csv")

comment_df = input_text_file.Comment
id_column = input_text_file.Id

# Remove special characters
print("Removing special characters from commit dataframe")
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–", "$", "ø", "å"]
comment_df = comment_df.str.replace('|'.join(map(re.escape, spec_chars)), '')
comment_df = comment_df.str.split().str.join(" ")

# Remove Ids
print("Removing IDs from commit dataframe")

comment_df = comment_df.apply(
    lambda row: re.sub(r'\w*\d\w*', '', row).strip())

# Stemming, Remove stop words
print("Tokenizing comments from commit dataframe")
print("Stemming, removing stop words and characters with a length of 1")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stemmer = PorterStemmer()
stop = stopwords.words('english')


def stem_remove_stop_words(row):
    place_holder = ""
    for word in w_tokenizer.tokenize(row):
        stem = stemmer.stem(word)
        if stem.lower() not in stop and len(stem) > 2:
            place_holder += stem + " "
    return place_holder


comment_df = comment_df.apply(lambda x: stem_remove_stop_words(x))

comment_df.to_csv(
    './data_cleaned/openstack/cleaned_data_openstack.csv', index=False)

# Bag of Words (1 - Gram)
# Remove words with frequency <= 3
print("Collecting most frequent words from clean commit data with frequency > 3")


def word_freq():
    wordfreq = {}
    to_delete = list()
    for index, row in comment_df.items():
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
    return wordfreq


wordfreq = word_freq()

print("Creating table for most frequent words given Comment column")

sentence_vectors = []
count = 1
for index, row in comment_df.items():
    sentece_tokens = nltk.word_tokenize(row)
    sent_vec = []
    for token in wordfreq:
        if token in sentece_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

word_frequency_openstack_df = pd.DataFrame(sentence_vectors, columns=wordfreq)

print("View most frequent word table in /part_III/data_cleaned/openstack/word_frequency_openstack.csv")
word_frequency_openstack_df = word_frequency_openstack_df.assign(
    Id=input_text_file.Id)
word_frequency_openstack_df.to_csv(
    './data_cleaned/openstack/word_frequency_openstack.csv', index=False)

print("View final training data with bag of words in /part_III/data_cleaned/openstack/training_data_final.csv")
training_data_final_df = pd.merge(left=input_training, right=word_frequency_openstack_df,
                                  how='left', left_on='Id', right_on='Id')
training_data_final_df.to_csv(
    './data_cleaned/openstack/training_data_final.csv', index=False)

print("View final testing data with bag of words in part_III/data_cleaned/openstack/testing_data_final.csv")
testing_data_final_df = pd.merge(left=input_test, right=word_frequency_openstack_df,
                                 how='left', left_on='Id', right_on='Id')
testing_data_final_df.to_csv(
    './data_cleaned/openstack/testing_data_final.csv', index=False)


train_target = training_data_final_df.isReq
training_data_final_df = training_data_final_df.drop('isReq', axis=1)
training_data_final_df = training_data_final_df.drop('Id', axis=1)

test_target = testing_data_final_df.isReq
testing_data_final_df = testing_data_final_df.drop('isReq', axis=1)
testing_data_final_df = testing_data_final_df.drop('Id', axis=1)

test_pred = DecisionTreeClassifier().fit(
    training_data_final_df, train_target).predict(testing_data_final_df)
print("DeicisionTreeClassifier", '\n', classification_report(
    test_target, test_pred, labels=[0, 1]))
