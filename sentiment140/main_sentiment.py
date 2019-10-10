import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer


cols = ['sentiment','id','date','query_string','user','text']
with open(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv") as f:
   df = pd.read_csv(f, header=None, names=cols)
   print(df)
   print(df.sentiment.value_counts())
   df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True) #inplace=True returns None inplace=False returns a copy of the object with the operation performed
   print(df[df.sentiment == 0].head(10)) #(0 = negative, 2 = neutral, 4 = positive). 50% of the data is with negative label, and another 50% with positive label.
   print(df[df.sentiment == 4].head(10)) #By looking at some entries for each class,
   # it seems like that all the negative class is from 0~799999th index, and the positive class entries start from 800000 to the end of the dataset.
   df['pre_clean_len'] = [len(t) for t in df.text]#The len() function returns the number of items in an object.
   print(df)

   data_dict = {
      'sentiment': {
         'type': df.sentiment.dtype,
         'description': 'sentiment class - 0:negative, 1:positive'
      },
      'text': {
         'type': df.text.dtype,
         'description': 'tweet text'
      },
      'pre_clean_len': {
         'type': df.pre_clean_len.dtype,
         'description': 'Length of the tweet before cleaning'
      },
      'dataset_shape': df.shape
   }
   pprint(data_dict)

   # fig, ax = plt.subplots(figsize=(5, 5))
   # plt.boxplot(df.pre_clean_len)
   # plt.show()

   print(df[df.pre_clean_len > 140].head(10))
   print (df.text[279])

   example1 = BeautifulSoup(df.text[279], 'lxml')
   print (example1.get_text())

   print(df.text[343])
   print(re.sub(r'@[A-Za-z0-9]+','',df.text[343]))

   print(df.text[0])
   print(re.sub('https?://[A-Za-z0-9./]+','',df.text[0]))

   print(df.text[226])
   testing = df.text[226]
   print(testing.replace(u"ï¿½", "?"))

   print(df.text[175])
   print(re.sub("[^a-zA-Z]", " ", df.text[175]))



   tok = WordPunctTokenizer()
   pat1 = r'@[A-Za-z0-9]+'
   pat2 = r'https?://[A-Za-z0-9./]+'
   combined_pat = r'|'.join((pat1, pat2))


   def tweet_cleaner(text):
      soup = BeautifulSoup(text, 'lxml')
      souped = soup.get_text()
      stripped = re.sub(combined_pat, '', souped)
      try:
         clean = stripped.decode("utf-8-sig").replace(u"ï¿½", "?")
      except:
         clean = stripped
      letters_only = re.sub("[^a-zA-Z]", " ", clean)
      lower_case = letters_only.lower()
      # During the letters_only process two lines above, it has created unnecessay white spaces,
      # I will tokenize and join together to remove unneccessary white spaces
      words = tok.tokenize(lower_case)
      return (" ".join(words)).strip()


   testing = df.text[:100]
   test_result = []
   for t in testing:
      test_result.append(tweet_cleaner(t))
   print(test_result)

   nums = [0, 400000, 800000, 1200000, 1600000]
   print ("Cleaning and parsing the tweets...\n")
   clean_tweet_texts = []
   for i in range(nums[0], nums[1]):
      if ((i + 1) % 10000 == 0):
         print ("Tweets %d of %d has been processed" % (i + 1, nums[1]))
      clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

      clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])
      clean_df['target'] = df.sentiment
      # print(clean_df.head())

      clean_df.to_csv('clean_tweet.csv', encoding='utf-8')
      csv = 'clean_tweet.csv'
      my_df = pd.read_csv(csv, index_col=0)
      print(my_df.head())



