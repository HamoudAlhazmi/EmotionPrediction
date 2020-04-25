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

class SentimentPredict:

    def __init__(self, sentiment):
        self.sentiment = sentiment
        self.class_model1 = None
        self.class_model2 = None

    def predict_model2(self):
        clf = load(path + '\\' + filename_model2)
        vectorizer = load(path + '\\' + filename_vectorizer_model2)

        text = [self.sentiment]
        vectors = vectorizer.transform(text)
        self.class_model2 = clf.predict(vectors)[0]
        return self.class_model2