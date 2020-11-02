import numpy as np
import csv as csv
import pandas as pd
import re
import nltk
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

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
print("Collecting most frequent words from clean commit data with frequency > 3")
vectorizer = CountVectorizer(stop_words="english", min_df=40)
frequent_words = vectorizer.fit_transform(comment_df)
print("Creating table for most frequent words given Comment column")


def bag_of_words():
    sentence_vectors = []
    for index, row in comment_df.items():
        sentece_tokens = nltk.word_tokenize(row)
        sent_vec = []
        for token in vectorizer.get_feature_names():
            if token in sentece_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)
    return sentence_vectors

word_frequency_openstack_df = pd.DataFrame(
    bag_of_words(), columns=vectorizer.get_feature_names())
word_frequency_openstack_df = word_frequency_openstack_df.assign(
    Id=input_text_file.Id)
training_data_final_df = pd.merge(left=input_training, right=word_frequency_openstack_df,
                                  how='left', left_on='Id', right_on='Id')
testing_data_final_df = pd.merge(left=input_test, right=word_frequency_openstack_df,
                                 how='left', left_on='Id', right_on='Id')
                                 
train_target = training_data_final_df.isReq
training_data_final_df = training_data_final_df.drop('isReq', axis=1)
training_data_final_df = training_data_final_df.drop('Id', axis=1)

test_target = testing_data_final_df.isReq
testing_data_final_df = testing_data_final_df.drop('isReq', axis=1)
testing_data_final_df = testing_data_final_df.drop('Id', axis=1)

word_frequency_openstack_df.to_csv(
    './data_cleaned/openstack/word_frequency_openstack.csv', index=False)
print("View most frequent word table in /part_III/data_cleaned/openstack/word_frequency_openstack.csv")
training_data_final_df.to_csv(
    './data_cleaned/openstack/training_data_final.csv', index=False)
print("View final training data with bag of words in /part_III/data_cleaned/openstack/training_data_final.csv")
testing_data_final_df.to_csv(
    './data_cleaned/openstack/testing_data_final.csv', index=False)
print("View final testing data with bag of words in part_III/data_cleaned/openstack/testing_data_final.csv")

test_pred = DecisionTreeClassifier().fit(
    training_data_final_df, train_target).predict(testing_data_final_df)
print("DeicisionTreeClassifier", '\n', classification_report(
    test_target, test_pred, labels=[0, 1]))
