'''
scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
- tokenizing: it is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful meanings
 elements called tokens. it converts a string like "My favorite color is blue" to a list of array like ["My", "favorite", "color", "is", "blue"] by using split() function.
- counting: method counts how many times an element has occurred in a list and returns it.
- normalizing: it works on the rows not columns
'''
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer implements both tokenization and occurrence counting in a single class
# The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.
# feature_extraction.text.CountVectorizer Convert a collection of text documents to a matrix of token counts.
vectorizer = CountVectorizer()
# print(vectorizer)

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)# .fit_transform method applies to feature extraction objects. the fit part applies
# to the feature extracture itself: it determines what features it will base future transformations on. The transform
# is what it takes the data and return some transformed data back at you.
print(X)
analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze'])

vectorizer.get_feature_names() == (
    ['and', 'document', 'first', 'is', 'one',
     'second', 'the', 'third', 'this'])
print(X.toarray())


