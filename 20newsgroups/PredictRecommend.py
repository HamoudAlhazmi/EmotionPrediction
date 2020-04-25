import feedparser

import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import pickle
from joblib import dump, load

path = r'PUT_YOUR_PATH_HERE'

filename_model2 = 'model2.joblib'
filename_vectorizer_model2 = 'vectorizer_model2.pickle'

filename_model2_class_dataframe = 'Model2_Class.csv'
filename_article_database = 'article_database.csv'

def predict(sentiment, path, filename_clf, filename_vectorizer):
    clf = load(path + '\\' + filename_clf)
    vectorizer = load(path + '\\' + filename_vectorizer)

    text = [sentiment]
    vectors = vectorizer.transform(text)
    predicted = clf.predict(vectors)[0]
    return predicted

def import_csv_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_csv(path)
    return df

def get_recommendation(predicted, path, filename_categories, filename_article_database):
    df_categories = import_csv_to_Dataframe(path, filename_categories)
    df_database = import_csv_to_Dataframe(path, filename_article_database)

    article_id = df_categories.iloc[predicted, 2]
    recommendation_article = df_database.iloc[int(article_id)].to_numpy()

    return recommendation_article

class Sentiment:

    def __init__(self, sentiment):
        self.sentiment = sentiment
        self.class_model1 = None
        self.class_model2 = predict(sentiment, path, filename_model2, filename_vectorizer_model2)

        recommendation_article = get_recommendation(self.class_model2, path, filename_model2_class_dataframe, filename_article_database)
        self.title = recommendation_article[0]
        self.description = recommendation_article[1]
        self.author = recommendation_article[2]
        self.category = recommendation_article[3]
        self.source = recommendation_article[4]
        self.link = recommendation_article[5]