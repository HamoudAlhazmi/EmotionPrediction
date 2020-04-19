#this .py will retrieve the latest news articles using RSS feeds python.
#the problem is that the model's accuracy is 82 with LinearSVC and we have to increase it to get better predicitons.
#We have been trying various ways to improve the model's accuracy but with no use.
import feedparser

#imports numpy and pandas
import numpy as np
import pandas as pd

import logging
from time import time
import sys

#imports sklearn libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# ----------------------------Calvin rss feeds----------------------

vectorizer = TfidfVectorizer(stop_words='english')
shuffle = True
remove = ['footers','quotes']

line_length = 120

#fetch train dataset and test dataset from 20newsgroups
#each subsets are shuffled and got their headers, footers, and quotes removed
newsgroup_train = fetch_20newsgroups(subset='train',
                                     shuffle=shuffle,
                                     remove=remove)
newsgroup_test = fetch_20newsgroups(subset='test',
                                    shuffle=shuffle,
                                    remove=remove)

#vectorize the datasets using TFIDF
X_train = vectorizer.fit_transform(newsgroup_train.data)
y_train = newsgroup_train.target
X_test = vectorizer.transform(newsgroup_test.data)

def print_line():
    print("="*line_length)

def model_header(model_name):
    print('')
    print_line()
    print(model_name)
    print_line()
    print('')
    return 0

def model_footer():
    print_line()

def custom_test_vectorizer(doc):
    vectors = vectorizer.transform(doc)
    return vectors

def custom_test(clf,doc):
    predicted = clf.predict(custom_test_vectorizer(doc))

    for doc, category in zip(doc, predicted):
        print('%r => %s' % (doc, newsgroup_train.target_names[category]))
    return predicted

def show_top10(classifier, vectorizer, categories):
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)
    print("Top 10 keyword per class:")
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s:\t\t %s" % (category, " ".join(feature_names[top10])))

def model_benchmark(model_name,text_clf):
    model_header(model_name)

    print("Model specification:")
    print(text_clf)

    t0 = time()
    text_clf.fit(X_train,y_train)
    train_time = time() - t0
    print("Train time: %0.3fs" % train_time)

    t0 = time()
    test_pred = text_clf.predict(X_test)
    test_time = time() - t0
    print("Test time:  %0.3fs" % test_time)

    print('')

    print("Classification Report:")
    print(metrics.classification_report(newsgroup_test.target, test_pred,
                                        target_names=newsgroup_test.target_names))

    show_top10(text_clf,vectorizer,newsgroup_train.target_names)

    print('')

    model_footer()

    return text_clf

final_model = model_benchmark('LinearSVC',
                              LinearSVC(loss='squared_hinge',
                              penalty='l2', dual=False,
                              tol=1e-3))

title_list = []
desc_list = []
link_list = []

category_list = [None] * 20

title_desc_list = []
n = 10

def filter_and_store(predicted):
    for idx, i in enumerate(predicted):
        if category_list[i] == None:
            category_list[i] = idx
    return 0

def predict_with_model_2(clf, text):
    predicted = clf.predict(vectorizer.transform(text))

    #for text, category in zip(text, predicted):
        #print('%r => %s' % (text, newsgroup_train.target_names[category]))

    return predicted

def show_filtered_results():
    for i in range(20):
        print_line()
        print(i)
        print(newsgroup_train.target_names[i])
        if category_list[i] == None:
            print('No article yet.')
        else:
            print('Title:      \t%r' % (title_list[category_list[i]]))
            print('Description:\t%r' % (desc_list[category_list[i]]))
            print('URL:        \t%r' % (link_list[category_list[i]]))
    return 0

def rss_feed(feed,n):
    for i in range(n):
        title_list.append(feed.entries[i].title)
        desc_list.append(feed.entries[i].description)
        link_list.append(feed.entries[i].link)
        title_desc_list.append(feed.entries[i].title + ". " + feed.entries[i].description)
    return 0

feed = feedparser.parse('http://www.abc.net.au/news/feed/2942460/rss.xml')
rss_feed(feed,25)

feed = feedparser.parse('http://www.9news.com.au/rss')
rss_feed(feed,20)

feed = feedparser.parse('http://feeds.smh.com.au/rssheadlines/top.xml')
rss_feed(feed,30)

feed = feedparser.parse('https://www.news.com.au/feed/')
rss_feed(feed,10)

predicted = predict_with_model_2(final_model,title_desc_list)

filter_and_store(predicted)

show_filtered_results()


