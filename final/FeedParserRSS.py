from Global import Global

import feedparser

import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import pickle
from joblib import dump, load

#####################################################################################
#Initial variables
#####################################################################################

const = Global()

feed_source = [['https://www.abc.net.au/science/news/topic/tech/tech.xml', 'ABC Australia'],
               ['http://www.9news.com.au/rss', '9NEWS'],
               ['http://www.dailytelegraph.com.au/entertainment/sydney-confidential/rss', 'Daily Telegraph'],
               ['http://feeds.smh.com.au/rssheadlines/top.xml', 'SMH Australian Breaking News'],
               ['https://www.news.com.au/feed/', 'News.com.au']]

no_of_articles_per_source = 40

df_articles = pd.DataFrame(columns=['Title',
                                    'Description',
                                    'Author',
                                    'Category',
                                    'Source',
                                    'Link',
                                    'Class_Model1',
                                    'Class_Model2'])

df_recommendation = pd.DataFrame(columns=['Class_Model1',
                                          'Class_Model2',
                                          'Article_ID'])

#####################################################################################
#Import and export functions
#####################################################################################

def load_model(path, filename):
    path = path + '\\' + filename
    clf = load(path)
    #print('Loaded %r from %r' % (filename, path))
    return clf

def load_feature_extraction(path, filename):
    path = path + '\\' + filename
    vectorizer = load(path)
    return vectorizer

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path, index=False)

    print('Dataframe exported to ' + path)
    return 0

def import_csv_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_csv(path)
    print('Dataframe imported from ' + path)
    return df

def import_model2_categories(path,filename):
    df = pd.DataFrame
    df = import_csv_to_Dataframe(path, filename)
    df.drop([0,1])
    return df

#####################################################################################
#RSS Feed Functions
#####################################################################################

def remove_html_variables(text):
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    return text


def rss_feed_to_dataframe(feed, df, no_of_article, source):
    count = 0
    for i in range(no_of_article):
        article_tuple = [None] * 8

        try:
            article_tuple[0] = remove_html_variables(feed.entries[i].title)
        except AttributeError as error:
            article_tuple[0] = None
        except IndexError as error:
            break

        try:
            article_tuple[1] = remove_html_variables(feed.entries[i].description)
        except AttributeError as error:
            article_tuple[1] = None

        try:
            article_tuple[2] = feed.entries[i].author
        except AttributeError as error:
            article_tuple[2] = None

        try:
            article_tuple[3] = feed.entries[i].category
        except AttributeError as error:
            article_tuple[3] = None

        article_tuple[4] = source

        try:
            article_tuple[5] = feed.entries[i].link
        except AttributeError as error:
            article_tuple[5] = None

        article_tuple[6] = None
        article_tuple[7] = None

        df = df.append({'Title': article_tuple[0],
                        'Description': article_tuple[1],
                        'Author': article_tuple[2],
                        'Category': article_tuple[3],
                        'Source': article_tuple[4],
                        'Link': article_tuple[5],
                        'Class_Model1': article_tuple[6],
                        'Class_Model2': article_tuple[7]},
                       ignore_index=True)
        count = count + 1

    return df

def count_articles(feed, max_article):
    count = 0
    test = None

    for i in range(max_article):
        try:
            test = feed.entries[i].title
        except IndexError as error:
            break

        count = count + 1

    return count

def rss_feed(feed_source, df, no_of_articles_per_source):
    for idx, source in enumerate(feed_source):
        feed = feedparser.parse(source[0])
        df = rss_feed_to_dataframe(feed, df, no_of_articles_per_source, source[1])
        count = count_articles(feed,no_of_articles_per_source)
        print("Stored %i articles from %r source" % (count, source[1]))
    return df

#####################################################################################
#Predict Functions
#####################################################################################

def article_preprocess(df):
    length = len(df)

    title_desc_list = [None] * length
    for i in range(length):
        title_desc_list[i] = df.iloc[i, 0] + '. ' + df.iloc[i, 1]

    return title_desc_list


def predict_article(df, clf, vectorizer):
    text = article_preprocess(df)

    vectors = vectorizer.transform(text)
    predicted = clf.predict(vectors)

    return predicted

def set_recommendation_articles_model2(df_recommendation,df_article):
    classes_model2 = df_article['Class_Model2'].to_numpy()

    for idx, i in enumerate(classes_model2):
        if math.isnan(df_recommendation.iloc[i, 2]):
            df_recommendation.iloc[i, 2] = idx
            df_recommendation.iloc[i, 3] = df_article.iloc[idx, 0]

    return df_recommendation


def set_recommendation_df(df_recommendation, df_article):

    #print(df_article.info())

    for i in range(20):
        for j in [0, 4]:
            df_temp = df_article[df_article.Class_Model2 == i]
            df_temp = df_temp[df_temp.Class_Model1 == j]

            try:
                df_recommendation = df_recommendation.append({'Class_Model1': j,
                                                              'Class_Model2': i,
                                                              'Article_ID': df_temp.index[0]},
                                                             ignore_index=True)
            except IndexError as error:
                df_recommendation = df_recommendation.append({'Class_Model1': j,
                                                              'Class_Model2': i,
                                                              'Article_ID': None},
                                                             ignore_index=True)
    return df_recommendation

#####################################################################################
#Run the entire code
#####################################################################################

def main(df_articles, df_recommendation):
    df_recommendation_article_by_model2 = import_model2_categories(const.path, const.filename_model2_class_dataframe)

    df_articles = rss_feed(feed_source, df_articles, no_of_articles_per_source)

    df_articles['Class_Model1'] = predict_article(df_articles, load_model(const.path, const.filename_model1),
                                                  load_feature_extraction(const.path, const.filename_vectorizer_model1))

    df_articles['Class_Model2'] = predict_article(df_articles, load_model(const.path, const.filename_model2),
                                                  load_feature_extraction(const.path, const.filename_vectorizer_model2))

    df_recommendation_article_by_model2 = set_recommendation_articles_model2(df_recommendation_article_by_model2,
                                                                             df_articles)

    df_recommendation = set_recommendation_df(df_recommendation, df_articles)

    export_Dataframe_to_csv(df_articles, const.path, const.filename_article_database)
    export_Dataframe_to_csv(df_recommendation_article_by_model2, const.path, const.filename_model2_class_dataframe)
    export_Dataframe_to_csv(df_recommendation, const.path, const.filename_article_recommendation)

main(df_articles, df_recommendation)