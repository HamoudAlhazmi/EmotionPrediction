'''
# project aim is to build a machine learning model that can extract the human emotions from text by using sentiment140.
download the dataset from here: http://help.sentiment140.com/for-students
 Tutorial source https://mathpn.github.io/sentiment-classification-twitter-part1/
'''
# Import libraries
import re # Regular Expression
import pandas as pd # Present data that is suitable for datta analysis via its series and dataframe data structures.
import numpy as np # It provides a high-performance multidimensional array object
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import feedparser

import pickle
from joblib import dump, load

final_model = (LogisticRegression(C = 1, penalty = 'l1', solver = 'liblinear',\
                multi_class = 'ovr', random_state = 42),)
model_name = ["LogisticRegression"]


# Load the data
print("Loading data...")
start_time = time.time() # This function is used to count the number of seconds elapsed since the epoch.

data = pd.read_csv(r"*PATH*", #download abovementioned 140 sentiment from website and add the path here between ""
                   encoding='ISO-8859-1', header=None)



line_length = 120
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


# data = data[0:800000]
print("The data shape is {}".format(data.shape))
data.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'tweet']
print(data.head())
print("Loading data has completed in {}s!\n".format(time.time() - start_time))

# Directly influences the model’s performance
def preprocess_tweets(tweet):
    return tweet

    # Detect ALLCAPS words
    tweet = re.sub(r"([A-Z]+\s?[A-Z]+[^a-z0-9\W]\b)", r"\1 <ALLCAPS> ", tweet)
    # Remove URLs
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '<URL> ', tweet)
    # Separate words that are joined by / (e.g. black/brown)
    tweet = re.sub(r"/", " / ", tweet)
    # Remove user mentions
    tweet = re.sub('@[^\s]+', "<USER>", tweet)
    # Remove all special symbols
    tweet = re.sub('[^A-Za-z0-9<>/.!,?\s]+', '', tweet)
    # Detect puncutation repetition
    tweet = re.sub('(([!])\\2+)', '! <REPEAT> ', tweet)
    tweet = re.sub('(([?])\\2+)', '? <REPEAT> ', tweet)
    tweet = re.sub('(([.])\\2+)', '. <REPEAT> ', tweet)
    # Remove hashtags
    tweet = re.sub(r'#([^\s]+)', r'<HASHTAG> \1', tweet)
    # Detect word elongation (e.g. heyyyyyy)
    tweet = re.sub(r'(.)\1{2,}\b', r'\1 <ELONG> ', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1)', tweet)
    # Expand english contractions
    tweet = re.sub(r"'ll", " will", tweet)
    tweet = re.sub(r"'s", " is", tweet)
    tweet = re.sub(r"'d", " d", tweet)  # Would/Had ambiguity
    tweet = re.sub(r"'re", " are", tweet)
    tweet = re.sub(r"didn't", "did not", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"can't", "cannot", tweet)
    tweet = re.sub(r"doesn't", "does not", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"'ve", " have", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    # Remove extra spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Lower case
    tweet = tweet.lower()

    return tweet


print("Apply train_test_split...")
start_time = time.time()
train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)
print("train_test_split has completed in {}s!\n".format(time.time() - start_time))

print("The train_data shape after splitting is {}".format(train_data.shape))
print("The test_data shape after splitting is {}".format(test_data.shape))

print("Apply preprocess_tweets...")
start_time = time.time()

sentiment = np.array(data['sentiment'])
tweets = np.array(data['tweet'].apply(preprocess_tweets))

sentiment_train = np.array(train_data['sentiment'])
tweets_train = np.array(train_data['tweet'].apply(preprocess_tweets))

sentiment_test = np.array(test_data['sentiment'])
tweets_test = np.array(test_data['tweet'].apply(preprocess_tweets))
print("preprocess_tweets has completed in {}s!\n".format(time.time() - start_time))

print("Do feature extraction...")
start_time = time.time()
vectorizer = TfidfVectorizer(min_df=75)
vectorizer.fit(tweets)
tweets_bow_train = vectorizer.transform(tweets_train)
tweets_bow_test = vectorizer.transform(tweets_test)
print("Feature extraction has been completed in {}s!\n".format(time.time() - start_time))


print("Create models...")
for cls_name, clf, in zip(model_name, final_model):
    print("Training {} ...".format(cls_name))
    start_time = time.time()
    clf.fit(tweets_bow_train, sentiment_train)
    print("Training models has been completed in {}s!\n".format(time.time() - start_time))

    print("Testing code...")
    start_time = time.time()
    pred1 = clf.predict(tweets_bow_test)
    pos_prob1 = clf.predict_proba(tweets_bow_test)[:, 1]
    auc1 = roc_auc_score(sentiment_test, pos_prob1)
    f11 = f1_score(sentiment_test, pred1, pos_label=4)
    print("Model 1: AUC {} F1 {}".format(auc1, f11))
    print("Testing code has completed in {}s!\n".format(time.time() - start_time))
    final_model = clf

# ------------------Check accuracy of news articles with rss feeds------------------

perdictions = final_model.predict(tweets_bow_test)

print_line()

path1 = r"*PATH*"
filename_model1 = 'model1.joblib'
filename_model1_class_dataframe = 'Model1_Class.csv'
filename_vectorizer_model1 = 'vectorizer_model1.pickle'#this is to save the vectorizer


def store_clf(clf, path, filename):
    path = path + '\\' + filename
    dump(clf, path)
    #print('Stored %r to %r' % (filename, path))

def store_vectorizer(vectorizer, path, filename):
    path = path + '\\' + filename
    dump(vectorizer, path)

def store_model1_and_vectorizer(clf, vectorizer, path, filename_model1, filename_vectorizer_model1):
    store_clf(clf, path, filename_model1)
    store_vectorizer(vectorizer, path, filename_vectorizer_model1)

store_model1_and_vectorizer(final_model, vectorizer, path1, filename_model1, filename_vectorizer_model1)


def set_recommendation_Dataframe_model1(df):
    df = df.append({'Class_No' : 0,
                    'Class_Name' : 'Negative',
                    'Article_ID' : None,
                    'Article_Title': None},
                   ignore_index=True)
    df = df.append({'Class_No': 2,
                    'Class_Name': 'Neutral',
                    'Article_ID': None,
                    'Article_Title': None},
                   ignore_index=True)
    df = df.append({'Class_No': 4,
                    'Class_Name': 'Positive',
                    'Article_ID': None,
                    'Article_Title': None},
                   ignore_index=True)
    return df

def process_Dataframe(df, path, filename):
    df = set_recommendation_Dataframe_model1(df)
    export_Dataframe_to_csv(df, path, filename)

df_recommendation_article_by_model1 = pd.DataFrame(columns=['Class_No',
                                                            'Class_Name',
                                                            'Article_ID',
                                                            'Article_Title'])

df_recommendation_article_by_model1 = set_recommendation_Dataframe_model1(df_recommendation_article_by_model1)

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path, index=False)

    return 0

export_Dataframe_to_csv(df_recommendation_article_by_model1, path1, filename_model1_class_dataframe)


