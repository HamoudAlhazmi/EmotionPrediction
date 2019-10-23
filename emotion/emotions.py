'''
 Tutorial source https://mathpn.github.io/sentiment-classification-twitter-part1/
'''
# Import libraries
import re # Regular Expression
import pandas as pd # Present data that is suitable for datta analysis via its series and dataframe data structures. it has variety of utilities to perform I/O operations in a seamless manner
import numpy as np # It provides a high-performance multidimensional array object
from matplotlib import pyplot as plt
import time
import texttable as tt
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Converts a collection of raw documents to a matrix of TF-IDF features
import plotly.graph_objects as go
import itertools




# Define the models
models = (MultinomialNB(alpha=1.0, fit_prior=True),
          RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
          DecisionTreeClassifier(max_depth=5),
          LogisticRegression(C = 1, penalty = 'l1', solver = 'liblinear',\
                multi_class = 'ovr', random_state = 42),)
models_name = ["MultinomialNB", "RandomForestClassifier", "DecisionTreeClassifier", "LogisticRegression"]

# Load the data
print("Loading data...")
start_time = time.time() # This function is used to count the number of seconds elapsed since the epoch.

data = pd.read_csv(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv",
                   encoding='ISO-8859-1', header=None)

# data = data[0:800000]
print("The data shape is {}".format(data.shape))
data.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'tweet']

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
for cls_name, clf, in zip(models_name, models):
    print("Training {} ...".format(cls_name))
    start_time = time.time()
    clf.fit(tweets_bow_train, sentiment_train)
    print("Training models has been completed in {}s!\n".format(time.time() - start_time))

#for i in range(1, 5):
    print("Testing code...")
    start_time = time.time()
    pred1 = clf.predict(tweets_bow_test)
    pos_prob1 = clf.predict_proba(tweets_bow_test)[:, 1]
    auc1 = roc_auc_score(sentiment_test, pos_prob1)
    f11 = f1_score(sentiment_test, pred1, pos_label=4)
    print("Model 1: AUC {} F1 {}".format(auc1, f11))
    print("Testing code has completed in {}s!\n".format(time.time() - start_time))


# unit_costs = ["{}"
#                   .format(auc1)]
# costs = ["Model 1: F1 {}"
#                     .format(itertools.repeat(float(f11(range(4)))))]

# helloworld = [*(auc1)]

# for i in range(float[auc1]):

# dict = {auc1}
# type(dict)
line = {auc1}
type(line)
#
# dict[1]

tab = tt.Texttable()
headings = ['Names','Parameters','AUC','Unit_Costs']
tab.header(headings)

names = ['bar', 'chocolate', 'chips']
weights = [0.05, 0.1, 0.25]

#auc1 = roc_auc_score(sentiment_test, pos_prob1)

for row in zip(models,models_name, models_name, line):
        tab.add_row(row)

s = tab.draw()
print (s)


#
# titles = ('KNN 6',
#           'KNN 30',
#           'KNN 4',
#           'KNN 3',
#           'KNN 2',
#           'KNN 1')
#
# def make_meshgrid(x, y, h=.02):
#     """Create a mesh of points to plot in
#
#     Parameters
#
#     x: data to base x-axis meshgrid on
#     y: data to base y-axis meshgrid on
#     h: stepsize for meshgrid, optional
#     Returns
#
#     xx, yy : ndarray
#     """
#     x_min, x_max = x.min() + 1, x.max() - 1
#     y_min, y_max = y.min() + 1, y.max() - 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy
#
#
#
# def plot_contours(ax, clf1, xx, yy, **params):
#
#     Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out
#
# fig, sub = plt.subplots(2, 2, figsize=(12, 15))
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
#
# X0, X1 = tweets_bow_train[:, 0], tweets_bow_train[:, 1]
# xx, yy = make_meshgrid(X0, X1)
#
# for clf1, title, ax in zip(models, titles, sub.flatten()):
#     #X_LVQ = clf.weights
#     #y_LVQ = clf.label_weights
#     plot_contours(ax, clf1, xx, yy,
#                   cmap=plt.cm.coolwarm, alpha=0.8)
#
#     ax.scatter(X0, X1, c=auc1, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#
#     # ax.scatter(X_LVQ[:, 0], X_LVQ[:, 1], c=y_LVQ,
#     #            cmap=plt.cm.coolwarm, s=50, marker='^', edgecolors='k')
#
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel('Sepal length')
#     ax.set_ylabel('Sepal width')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(title)
# plt.show()
#





















# tab = tt.Texttable()
# headings = ['Names','Parameters','AUC','Unit_Costs']
# tab.header(headings)
# names = ['bar', 'chocolate', 'chips']
# weights = [0.05, 0.1, 0.25]
#
# # unit_costs = [40.0, 50.0, 12.0]
# # costs = ()
#
# for row in zip(models_name,models, unit_costs, costs):
#     tab.add_row(row)
#
# s = tab.draw()
# print (s)






# unit_costs = ["Model 1: AUC {blc}" .format(blc=auc1)]
# unit_costs1 = ["Model 1: AUC {:f}" .format(auc1)]
# print("Testing code has completed in {}s!\n".format(time.time() - start_time))
# # costs = ["Model 1: F1 {}"
# #                     .format(f11)]
#
# fig = go.Figure(data=[go.Table(header=dict(values=['Names','Parameters','AUC','F1 score']),
#                                cells=dict(values=[(models_name), (unit_costs), (unit_costs1)]))
#                       ])
# fig.show()







# tab = tt.Texttable()
# headings = ['Names','Parameters','AUC','F1 score']
# tab.header(headings)
# names = ['bar', 'chocolate', 'chips']
# weights = [0.05, 0.1, 0.25]
#
#
# unit_costs = ["Model 1: AUC {}"
#                   .format(auc1)]
# costs = ["Model 1: F1 {}"
#                     .format(f11)]
#
# for row in zip(models_name,models, unit_costs, costs):
#     #tab.add_rows(["auto"])
#     tab.add_row(row)
#     #tab.set_cols_align(["l", "r", "r", "r", "l"])
#     #tab.add_rows([["text", "float", "exp", "int", "auto"]
#
#     s = tab.draw()
#     print (s)
# # s = tab.draw()
# # print (s)




