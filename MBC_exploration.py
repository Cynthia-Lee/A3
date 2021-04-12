# Sources Used:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?fbclid=IwAR2e_uoplSxSBWON3XZ69JA1Fnck-SFFE42PUKAVPi_quhe8CQk4qUnReWQ
# https://scikit-learn.org/stable/modules/feature_extraction.html
# https://scikit-learn.org/stable/modules/preprocessing.html
# http://www.nltk.org/howto/stem.html

from nltk.corpus.reader.chasen import test
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk import word_tokenize
# nltk.download('punkt')
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV, RidgeClassifier, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Normalizer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, SelectFpr, SelectFromModel, SelectFwe, SequentialFeatureSelector, RFE, RFECV, VarianceThreshold, chi2, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sys


### Dataset ###
# Header
    # Consists of fields such as <From>, <Subject>, <Organization> and <Lines> fields.
    # The <lines> field includes the number of lines in the document body
# Body
    # The main body of the document. This is where you should extract features from.

def remove_header(text):
    index = text.find("\n\n")
    text = text[index:]
    return text

nltk_stop_words = stopwords.words('english')
my_stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))    
stemmer = SnowballStemmer("english")
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
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return [stemmer.stem(w) for w in word_tokenize(text) if w not in ENGLISH_STOP_WORDS]

def lemma_tokenizer(text):
    return [lemma.lemmatize(w) for w in word_tokenize(text) if w not in ENGLISH_STOP_WORDS]

# make sure that you preprocess your stop list to make sure that it is normalised like your tokens will be,
# and pass the list of normalised words as stop_words to the vectoriser.


def clf_pipe(twenty_train, vect, clf, select=None):
    X = twenty_train.data
    y = twenty_train.target
    pipe = []
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
    # predicted = text_clf.predict(docs_test)
    # info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    # result = ",".join(map(str,info[:-1]))
    # print("Test " + str(test_number) + ": " + result)
    return text_clf

def check_performance(twenty_evaluation, text_clf): #, scaler=None): #, encoder=None):
    # global test_number
    # test_number += 1
    predicted = text_clf.predict(twenty_evaluation.data) # twenty_evaluation.data
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    # print("Test " + str(test_number) + ": " + result)
    print(result)
    return result

def mbc_exploration(twenty_train, twenty_evaluation):
    docs_test = twenty_evaluation.data
    # print(len(twenty_train.data)) # 2170
    # print(len(twenty_evaluation.data)) # 721

    ### Preprocessing ###
    # remove headers
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

    # nltk_stop_words = stopwords.words('english')
    # my_stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))    
    # stemmer = SnowballStemmer("english")
    # lemma = WordNetLemmatizer()

    # -----

    global test_number
    test_number = 0

    # -----

    ### NB ###

    print("NB")
    clf = MultinomialNB()

    # Trying Feature Selection
    # vect = CountVectorizer()
    # select = SelectPercentile()
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # stop_words improve a bit and grid searched for max_df
    # vect = CountVectorizer(lowercase=True, stop_words=nltk_stop_words, max_df=0.5)
    # select = SelectPercentile()
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # test my_stop_words
    # vect = CountVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5)
    # select = SelectPercentile()
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # test nltk_stop_words with hyperparameters tuned
    # vect = CountVectorizer(lowercase=True, stop_words=nltk_stop_words, max_df=0.5, ngram_range=(1,2))
    # clf = MultinomialNB(alpha=0.001)
    # select = SelectPercentile(percentile=60)
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # test my_stop_words with hyperparameters tuned
    # vect = CountVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5, ngram_range=(1,2))
    # clf = MultinomialNB(alpha=0.001)
    # select = SelectPercentile(percentile=60)
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # try tokenize_and_lemma
    # vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # 2nd best configuration (no lemma)
    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5, ngram_range=(1,2))
    # clf = MultinomialNB(alpha=0.001)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # nb_c2 = (text_clf, check_performance(twenty_evaluation, text_clf))

    # best configuration (lemma)
    vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma, max_df=0.5, ngram_range=(1,3))
    clf = MultinomialNB(alpha=0.001)
    text_clf = clf_pipe(twenty_train, vect, clf)
    nb_c = (text_clf, check_performance(twenty_evaluation, text_clf))

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
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)
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
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # print("SelectKBest f_classif, k=all")
    # select = SelectKBest(f_classif, k='all')
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # print("SelectFromModel LinearSVC")
    # select = SelectFromModel(LinearSVC())
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # print("SelectFromModel LinearSVC penalty=11")
    # select = SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # print("SelectFromModel SGD penalty=11")
    # select = SelectFromModel(SGDClassifier())
    # text_clf = clf_pipe(twenty_train, vect, clf, select)
    # check_performance(twenty_evaluation, text_clf)

    # -----

    ### LR ###

    print("\nLR")
    test_number = 0

    vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)

    # clf = LogisticRegression(solver="lbfgs") # penalty=l2
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(solver="newton-cg") # penalty=l2
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(solver="sag") # penalty=l2
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(solver="saga")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(penalty='l2', solver='saga')
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(penalty='l2', solver='saga', tol=0.1)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(penalty='l1', tol=0.01, solver='saga', C=1000)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(penalty='l2', tol=0.01, solver='saga', C=1000)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # best configuration
    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)
    # clf = LogisticRegression(penalty='l1', tol=0.01, solver='saga', C=1000, max_iter=1000)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # lr_c1 = (text_clf, check_performance(twenty_evaluation, text_clf))

    # 2nd best configuration
    vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
    clf = LogisticRegression(penalty='l1', tol=0.01, solver='saga', C=1000, max_iter=1000)
    text_clf = clf_pipe(twenty_train, vect, clf)
    lr_c = (text_clf, check_performance(twenty_evaluation, text_clf))

    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)
    # clf = LogisticRegression(penalty='l2', tol=0.01, solver='saga', C=1000, max_iter=1000, multi_class="multinomial")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
    # clf = LogisticRegression(penalty='l1', tol=0.001, solver='saga', C=1000, max_iter=1000, multi_class="multinomial")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(penalty='l2', tol=0.001, solver='saga', C=1000, max_iter=1000)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, tol=0.01, C=50)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

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

    print("\nSVM")
    test_number = 0

    # SGDClassifier
    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    # clf = SGDClassifier(penalty="elasticnet")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)
    # clf = SGDClassifier(alpha=0.0001, penalty="elasticnet")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = SGDClassifier(penalty="elasticnet", l1_ratio=0.5)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # clf = SGDClassifier(penalty="l1")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # 2nd best configuration
    # clf = SGDClassifier(penalty="l2")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # svm_c2 = (text_clf, check_performance(twenty_evaluation, text_clf))

    # vect = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=my_stop_words, max_df=0.75)
    # clf = SGDClassifier(alpha=0.00001, penalty="l2")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # LinearSVC
    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    # clf = LinearSVC()
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # SVC
    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    # clf = SVC(C=10)
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # linear, poly, rbf, sigmoid, precomputed

    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    # clf = SVC(kernel="linear")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    # clf = SVC(kernel="poly")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    # best configuration
    vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    clf = SVC(kernel="sigmoid", C=10)
    text_clf = clf_pipe(twenty_train, vect, clf)
    svm_c = (text_clf, check_performance(twenty_evaluation, text_clf))
    # parameters = {
    #     # 'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
    #     # 'clf__max_iter': (10, 50, 80),
    #     # 'clf__penalty': ('l2', 'elasticnet'),
    #     # 'vect__max_df': (0.5, 0.75, 1.0),
    #     # 'vect__max_features': (None, 5000, 10000, 50000)
    #     'clf__C': [.001, .01, .1, 1, 10, 100, 1000]
    #  }
    # gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    # gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, ngram_range=(1,2))
    # clf = SVC(kernel="precomputer")
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

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

    # -----

    print("\nRF")
    test_number = 0

    # 2nd best configuration
    # vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words)
    # clf = RandomForestClassifier()
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # rf_c2 = (text_clf, check_performance(twenty_evaluation, text_clf))
    # parameters = {
    #     'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
    #     #'vect__max_df': [0.5, 0.7, 0.9],
    #     #'vect__sublinear_tf': [True, False],
    #     #'vect__min_df': [1, 5],
    # }
    # gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    # gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # best configuration
    vect = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, max_df=0.5, min_df=5, ngram_range=(1, 2))
    clf = RandomForestClassifier()
    text_clf = clf_pipe(twenty_train, vect, clf)
    rf_c =  (text_clf, check_performance(twenty_evaluation, text_clf))
    # number of trees, number of features
    # parameters = {
    #     # 'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
    #     # 'vect__max_df': [0.5, 0.7, 0.9],
    #     # 'vect__sublinear_tf': [True, False],
    #     # 'vect__min_df': [1, 5],
    #     # 'clf__criterion': ["gini", "entropy"],
    #     'clf__n_estimators': [50, 100, 200]
    # }
    # gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    # gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma)
    # clf = RandomForestClassifier()
    # text_clf = clf_pipe(twenty_train, vect, clf)
    # check_performance(twenty_evaluation, text_clf)

    return [("NB,LB", nb_c), ("LR,LSA", lr_c), ("SVM,SNSS", svm_c), ("RF,SN", rf_c)]

### Main class ###
if __name__ == '__main__':
    # (sys.argv[0]) # UB_BB.py
    trainset = (sys.argv[1])
    evalset = (sys.argv[2])
    output = (sys.argv[3])

    f = open(output, "w")

    categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']

    train_folder = ".\Selected 20NewsGroup\\" + trainset
    evaluation_folder = ".\Selected 20NewsGroup\\" + evalset
    twenty_train = load_files(train_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
    twenty_evaluation = load_files(evaluation_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')

    configurations = mbc_exploration(twenty_train, twenty_evaluation)

    for config in configurations:
        classification = config[0]
        c1 = (config[1])[1]
        f.write(classification + " " + c1 + "\n")
    f.close()


    