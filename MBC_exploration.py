# Sources Used:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?fbclid=IwAR2e_uoplSxSBWON3XZ69JA1Fnck-SFFE42PUKAVPi_quhe8CQk4qUnReWQ
# https://scikit-learn.org/stable/modules/feature_extraction.html
# https://scikit-learn.org/stable/modules/preprocessing.html
# http://www.nltk.org/howto/stem.html

from nltk.corpus.reader.chasen import test
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#
import re
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk import word_tokenize
#
nltk.download('punkt')
#
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV, RidgeClassifier, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Normalizer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, SelectFpr, SelectFromModel, SelectFwe, SequentialFeatureSelector, RFE, RFECV, VarianceThreshold, chi2, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

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
def remove_header(text):
    index = text.find("\n\n")
    text = text[index:]
    return text

for i in range(len(twenty_train.data)):
    text = twenty_train.data[i]
    twenty_train.data[i] = remove_header(text)
for i in range(len(twenty_evaluation.data)):
    text = twenty_evaluation.data[i]
    twenty_evaluation.data[i] = remove_header(text)

# clean text: remove numbers, punctuation, space, url, email (not implemented yet)

### Feature Extraction ###

# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features
# Feature Extractors
# 1. CountVectorizer - Uses the number of times each word was observed.
# 2. TFIDFVectorizer - Uses relative frequencies normalized by the inverse of the number of documents in which the word was observed
# Preprocessing
# a. lower case and filter out stopwords
# b. apply stemming

# TfidfVectorizer equivalent to CountVectorizer followed by TfidfTransformer

# stemming
nltk_stop_words = stopwords.words('english')
my_stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
stemmer = SnowballStemmer("english") # stemmer = SnowballStemmer("english", ignore_stopwords=True)
lemma = WordNetLemmatizer()

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token): # ignore non-letters
            filtered_tokens.append(token)
    #exclude stopwords from stemmed words
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in my_stop_words]
    return stems

def tokenize_and_lemma(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token): # ignore non-letters
            filtered_tokens.append(token)
    #exclude stopwords from lemma words
    stems = [lemma.lemmatize(t) for t in filtered_tokens if t not in my_stop_words]
    return stems

def stemming_tokenizer(text):
    return [stemmer.stem(w) for w in word_tokenize(text) if w not in ENGLISH_STOP_WORDS]

def lemma_tokenizer(text):
    return [lemma.lemmatize(w) for w in word_tokenize(text) if w not in ENGLISH_STOP_WORDS]

# make sure that you preprocess your stop list to make sure that it is normalised like your tokens will be,
# and pass the list of normalised words as stop_words to the vectoriser.

# -----

global test_number
test_number = 0

def check_performance(vect, clf, select=None): #, scaler=None): #, encoder=None):
    X = twenty_train.data
    y = twenty_train.target
    pipe = []
    global test_number
    test_number += 1
    # if (encoder):
    #     pipe.append(('encoder', encoder))
    #     # X = np.array(X).reshape(-1, 1)
    #     # pipe.append(('normal', Normalizer()))
    # if (scaler):
    #     pipe.append(('scaler', scaler))
    #     pipe.append(('pca', PCA(n_components=2)))
    # else:
    pipe.append(('vect', vect))
    if (select):
        pipe.append(('select', select))
    pipe.append(('clf', clf))
    text_clf = Pipeline(pipe)
    text_clf.fit(X, y)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    print("Test " + str(test_number) + ": " + result)
    return text_clf

# -----
'''
### NB ###

print("NB")
clf = MultinomialNB()

# Trying Feature Selection
# vect = CountVectorizer()
# select = SelectPercentile()
# text_clf = check_performance(vect, clf, select)

# stop_words improve a bit and grid searched for max_df
# vect = CountVectorizer(lowercase=True, stop_words=nltk_stop_words, max_df=0.5)
# select = SelectPercentile()
# text_clf = check_performance(vect, clf, select)

# test my_stop_words
# vect = CountVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5)
# select = SelectPercentile()
# text_clf = check_performance(vect, clf, select)

# test nltk_stop_words with hyperparameters tuned
# vect = CountVectorizer(lowercase=True, stop_words=nltk_stop_words, max_df=0.5, ngram_range=(1,2))
# clf = MultinomialNB(alpha=0.001)
# select = SelectPercentile(percentile=60)
# text_clf = check_performance(vect, clf, select)

# test my_stop_words with hyperparameters tuned
# vect = CountVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5, ngram_range=(1,2))
# clf = MultinomialNB(alpha=0.001)
# select = SelectPercentile(percentile=60)
# text_clf = check_performance(vect, clf, select)

# try tokenize_and_lemma
# vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
# text_clf = check_performance(vect, clf)

# 2nd best configuration (no lemma)
vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5, ngram_range=(1,2))
clf = MultinomialNB(alpha=0.001)
text_clf = check_performance(vect, clf)

# best configuration (lemma)
vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma, max_df=0.5, ngram_range=(1,3))
clf = MultinomialNB(alpha=0.001)
text_clf = check_performance(vect, clf)

# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3), (1, 4)],
#     #'vect__max_features': (None, 5000, 10000, 50000),
#     #'vect__max_df': [0.5, 0.7, 0.9],
#     #'vect__sublinear_tf': [True, False],
#     'vect__min_df': [1, 5],
#     'clf__alpha': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# Other tests:

# Feature_selections to try: SelectKBest, SelectFromModel
# score_func for classification: chi2, f_classif, mutual_info_classif

# print("SelectKBest chi, k=20")
# select = SelectKBest(chi2, k=20)
# text_clf = check_performance(vect, clf, select)
# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3), (1, 4), (4, 4)],
#     'vect__max_df': [0.5, 0.7, 0.9],
#     'select__k': [10, 20, 30, 40, 50, "all"],
#     'clf__alpha': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# print("SelectKBest chi, k=all")
# select = SelectKBest(chi2, k='all')
# text_clf = check_performance(vect, clf, select)

# print("SelectKBest f_classif, k=all")
# select = SelectKBest(f_classif, k='all')
# text_clf = check_performance(vect, clf, select)

# print("SelectFromModel LinearSVC")
# select = SelectFromModel(LinearSVC())
# text_clf = check_performance(vect, clf, select)

# print("SelectFromModel LinearSVC penalty=11")
# select = SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))
# text_clf = check_performance(vect, clf, select)

# print("SelectFromModel SGD penalty=11")
# select = SelectFromModel(SGDClassifier())
# text_clf = check_performance(vect, clf, select)
'''
# -----

### LR ###

print("\nLR")

test_number = 0
vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)

# clf = LogisticRegression(solver="lbfgs") # penalty=l2
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(solver="newton-cg") # penalty=l2
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(solver="sag") # penalty=l2
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(solver="saga")
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(penalty='l2', solver='saga')
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(penalty='l2', solver='saga', tol=0.1)
# text_clf = check_performance(vect, clf)

clf = LogisticRegression(penalty='l1', tol=0.01, solver='saga', C=1000)
text_clf = check_performance(vect, clf)

clf = LogisticRegression(penalty='l2', tol=0.01, solver='saga', C=1000)
text_clf = check_performance(vect, clf)

# best configuration
vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)
clf = LogisticRegression(penalty='l1', tol=0.01, solver='saga', C=1000, max_iter=1000)
text_clf = check_performance(vect, clf)

# 2nd best configuration
vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
clf = LogisticRegression(penalty='l1', tol=0.01, solver='saga', C=1000, max_iter=1000)
text_clf = check_performance(vect, clf)

# vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)
# clf = LogisticRegression(penalty='l2', tol=0.01, solver='saga', C=1000, max_iter=1000, multi_class="multinomial")
# text_clf = check_performance(vect, clf)

# vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
# clf = LogisticRegression(penalty='l1', tol=0.001, solver='saga', C=1000, max_iter=1000, multi_class="multinomial")
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(penalty='l2', tol=0.001, solver='saga', C=1000, max_iter=1000)
# text_clf = check_performance(vect, clf)

# clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, tol=0.01, C=50)
# text_clf = check_performance(vect, clf)

# parameters = {
#     'clf__max_iter': (10, 50, 80),
#     'vect__use_idf': (True, False),
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'vect__max_features': (None, 5000, 10000, 50000),
#     # 'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3)],
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# -----
'''
### SVM ###
print("\nSVM")

print("SGD")
vect = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, tokenizer=tokenize_and_lemma, max_df=0.75)
clf = SGDClassifier(alpha=0.00001, penalty="l2")
text_clf = check_performance(vect, clf)

vect = TfidfVectorizer(max_df=0.75, max_features=50000)
clf = SGDClassifier(alpha=9.9999999999999995e-07, penalty="elasticnet", max_iter=50)
text_clf = check_performance(vect, clf)

vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, max_features=50000)
clf = SGDClassifier(alpha=9.9999999999999995e-07, penalty="elasticnet", max_iter=50)
text_clf = check_performance(vect, clf)
# parameters = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'vect__max_features': (None, 5000, 10000, 50000),
#     'vect__ngram_range': ((1, 2), (1, 3)),
#     'clf__penalty': ('l2', 'elasticnet'),
#     # 'clf__max_iter': (10, 50, 80),
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

vect = TfidfVectorizer(ngram_range=(1, 3), max_df=0.75, max_features=None)
clf = SGDClassifier(alpha=9.9999999999999995e-07, penalty="elasticnet", max_iter=50)
text_clf = check_performance(vect, clf)

# vect = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, tokenizer=tokenize_and_lemma, max_df=0.75)
# clf = SGDClassifier(alpha=0.00001, penalty="l2")
# text_clf = check_performance(vect, clf)
# parameters = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'vect__max_features': (None, 5000, 10000, 50000),
#     # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     # 'tfidf__use_idf': (True, False),
#     # 'tfidf__norm': ('l1', 'l2'),
#     # 'clf__alpha': (0.00001, 0.000001),
#     'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
#     'clf__penalty': ('l2', 'elasticnet'),
#     'clf__max_iter': (10, 50, 80),
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# NB C4
# estimator = DecisionTreeClassifier()
# c = chi2 # chi2, f_classif, mutual_info_classif
# try_features = [SelectPercentile(c), SelectKBest(c), SelectFpr(c), SelectFromModel(estimator), SelectFwe(c), VarianceThreshold()]
# RFE(estimator), RFECV(estimator), SequentialFeatureSelector(estimator, cv=2)
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
# model = SelectFromModel(lsvc)
# try_features = [model]

# for feature in try_features:
#     print(feature)
#     text_clf = Pipeline([
#             ('vect', CountVectorizer(ngram_range=(1,2), lowercase=True, stop_words=my_stop_words)), # vector
#             ('tfidf', TfidfTransformer(use_idf=True)), # transformer
#             ('select', feature), # feature selection
#             ('clf', MultinomialNB(alpha=0.001)), # classifier
#         ])
#     text_clf.fit(twenty_train.data, twenty_train.target)
#     predicted = text_clf.predict(docs_test)
#     info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
#     result = ",".join(map(str,info[:-1]))
#     print("NB,C4," + result + "\n")

# parameters = {
#     #'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3), (1, 4), (4, 4)],
#     'select__percentile': (0, 100)
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

### More Hyperparameters Tuning ##

# -----

# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3), (1, 4), (4, 4)],
#     'clf__tol': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
'''