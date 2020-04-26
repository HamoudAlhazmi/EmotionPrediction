import PredictRecommend

input_sentiment = 'I love cars!'

sentiment = PredictRecommend.Sentiment(input_sentiment)

print(sentiment.title)
print(sentiment.description)
print(sentiment.source)
print(sentiment.link)