# Sources Used:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?fbclid=IwAR2e_uoplSxSBWON3XZ69JA1Fnck-SFFE42PUKAVPi_quhe8CQk4qUnReWQ
# https://scikit-learn.org/stable/modules/feature_extraction.html
# https://scikit-learn.org/stable/modules/preprocessing.html
# http://www.nltk.org/howto/stem.html

from nltk.stem.porter import PorterStemmer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
#
nltk.download('punkt')
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

### Dataset ###
# Header
    # Consists of fields such as <From>, <Subject>, <Organization> and <Lines> fields.
    # The <lines> field includes the number of lines in the document body
# Body
    # The main body of the document. This is where you should extract features from.

### Loading the 20 newsgroups dataset ### 
categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']
train_folder = ".\Selected 20NewsGroup\Training"
evaluation_folder = ".\Selected 20NewsGroup\Evaluation"

twenty_train = load_files(train_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
twenty_evaluation = load_files(evaluation_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
docs_test = twenty_evaluation.data
# print(len(twenty_train.data)) # 2170
# print(len(twenty_evaluation.data)) # 721

### Preprocessing ###

# remove headers
# def remove_header(text):
#     index = text.find("\n\n")
#     text = text[index:]

for i in range(len(twenty_train.data)):
    text = twenty_train.data[i]
    index = text.find("\n\n")
    twenty_train.data[i] = text[index:]
for i in range(len(twenty_evaluation.data)):
    text = twenty_evaluation.data[i]
    index = text.find("\n\n")
    twenty_evaluation.data[i] = text[index:]

# clean text
# remove numbers, punctuation, space, url, email

# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features
# Feature Extractors
# 1. CountVectorizer - Uses the number of times each word was observed.
# 2. TFIDFVectorizer - Uses relative frequencies normalized by the inverse of the number of documents in which the word was observed
# Preprocessing
# a. lower case and filter out stopwords
# b. apply stemming

# TfidfVectorizer equivalent to CountVectorizer followed by TfidfTransformer
# lowercase=True (default)
# stop_words
# token_pattern
    # The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).
# ngram_range
# max_df
# min_df

# stemming
# def stemming_tokenizer(text):
#     result = []
#     stemmer = SnowballStemmer("english")
#     # exclude stopwords from stemmed words
#     for w in text.split(" "):
#         if (w.lower() not in stopwords.words('english')):
#             result.append(stemmer.stem(w))
#         else:
#             result.append(w)
#     return result

def stemming_tokenizer(text):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w) for w in word_tokenize(text)]

# make sure that you preprocess your stop list to make sure that it is normalised like your tokens will be,
# and pass the list of normalised words as stop_words to the vectoriser.

# print(stemming_tokenizer("I didn't want to go fishing to get fishes"))
my_stop_words = stopwords.words('english')
# exclude nltk stopwords, punctuation, integers

# -----

# NB C1
text_clf = Pipeline([
    ('vect', TfidfVectorizer(lowercase=True, stop_words=my_stop_words)), # vector
    ('clf', MultinomialNB(alpha=1.0)), # classifier
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
result = ",".join(map(str,info[:-1]))
print("NB,C1," + result + "\n")

# # NB C1.5
# print("Stopwords and Stemming")
# text_clf = Pipeline([
#     ('vect', TfidfVectorizer(lowercase=True, tokenizer=stemming_tokenizer, stop_words=my_stop_words)), # vector
#     ('clf', MultinomialNB(alpha=1.0)), # classifier
# ])
# text_clf.fit(twenty_train.data, twenty_train.target)
# predicted = text_clf.predict(docs_test)
# info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
# result = ",".join(map(str,info[:-1]))
# print("NB,C1.5," + result + "\n")

# Feature selection

# Hyperparameters tuning

### Hyperparameters
# 1. Naive Bayes - no hyperparameters 
# 2. Logistic Regression - Regularization constant, num iterations
# 3. SVM - Regularization constant, Linear, polynomial or RBF kernels.
# 4. RandomForest - Number of trees and number of features to consider.

# NB C2
text_clf = Pipeline([
    ('vect', TfidfVectorizer(lowercase=True, stop_words=my_stop_words)), # vector
    ('clf', MultinomialNB(alpha=0.5)), # classifier
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
result = ",".join(map(str,info[:-1]))
print("NB,C2," + result + "\n")


'''
### Feature Selection
# L1, L2 or Lasso regularizers

# https://scikit-learn.org/stable/modules/feature_selection.html

# make sure that you preprocess your stop list to make sure that it is normalised like your tokens will be,
# and pass the list of normalised words as stop_words to the vectoriser.

class StemTokenizer:
    def __init__(self):
        self.ss = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.ss.stem(t) for t in word_tokenize(doc)]

stemmer = StemTokenizer()
stem_stop_words = stemmer(' '.join(nltk_stop_words)) # preprocess stop_words

### Tutorial: Extracting features from text files
# vectorizer

# unigram baseline (UB)
# count_vect = CountVectorizer(lowercase=True, stop_words=stem_stop_words, tokenizer=StemTokenizer(), analyzer='word') 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data) # matrix
print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'algorithm')) # counts of N-grams of words or consecutive characters

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# bigram baseline (BB)
# bigram_vectorizer = CountVectorizer(lowercase=True, stop_words=stem_stop_words, analyzer='word', ngram_range=(2,2))
# X_train_counts_bb = bigram_vectorizer.fit_transform(twenty_train.data)
# print(X_train_counts_bb.shape)

### Tutorial: Training a classifier
# classifier fit

# Naive Bayes (NB)
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

### Tutorial: Building a pipeline
# In order to make the vectorizer => transformer => classifier easier to work with, 
# scikit-learn provides a Pipeline class that behaves like a compound classifier:

# Naive Bayes (NB)
text_clf = Pipeline([
    ('vect', CountVectorizer()), # vector
    ('tfidf', TfidfTransformer()), # transformer
    ('clf', MultinomialNB()), # classifier
])

print(text_clf.fit(twenty_train.data, twenty_train.target)) # train the model

### Tutorial: Evaluation of the performance on the test set
evaluation_folder = ".\Selected 20NewsGroup\Evaluation"

twenty_test = load_files(evaluation_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

# support vector machine (SVM)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
        alpha=1e-3, random_state=42,
        max_iter=5, tol=None)),
])

print(text_clf.fit(twenty_train.data, twenty_train.target))
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))

### Tutorial: Parameter tuning using grid search
# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__use_idf': (True, False),
#     'clf__alpha': (1e-2, 1e-3),
# }

# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
# print(gs_clf.best_score_)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
'''