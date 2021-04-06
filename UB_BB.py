# Basic Comparison with Baselines - For all four methods (NB,LR, SVM, and RF), 
# you should run both unigram and bigram baselines.

# should run all 4 methods, each with 2 configurations.

# python UB_BB.py <trainset> <evalset> <output> <display_LC>

# <trainset> is the parent folder to your training data and
# <evalset> is the parent folder to your evaluation data
# <display_LC> is an option to display the learning curve (part b). `1' to show the plot, `0' to NOT show the plot. 
# <output> is the path to a comma-separated values file, which contains 8 lines corresponding to 8 runs. Note that all elements are UPPER-CASE letters.
    # Each line should be '<Classifcation Method>, <Configuration>, <Macro-Precision>, <Macro-Recall>, <F1-score>', 
    # evaluated on the evaluation data, not training data.

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

### Main class ###
if __name__ == '__main__':
    # (sys.argv[0]) # UB_BB.py
    # trainset = (sys.argv[1])
    # evalset = (sys.argv[2])
    # output = (sys.argv[3])
    # display_LC = (sys.argv[4]) # `1' to show the plot

    categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']
    train_folder = ".\Selected 20NewsGroup\Training"
    evaluation_folder = ".\Selected 20NewsGroup\Evaluation"

    twenty_train = load_files(train_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
    twenty_evaluation = load_files(evaluation_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')

    # -----

    # NB UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', MultinomialNB()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    # print(np.mean(predicted == twenty_evaluation.target)) # accuracy
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    # precision, # recall, fbeta_score, support
    print("NB,UB,", result)

    # NB BB
    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', MultinomialNB()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    # print(np.mean(predicted == twenty_evaluation.target)) # accuracy
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("NB,BB,", result)

    # -----

    # LR UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', LogisticRegression()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("LR,UB,", result)

    # LR BB
    text_clf = Pipeline([
        ('vect',CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', LogisticRegression()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("LR,BB,", result)

    # -----

    # SVM UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', SGDClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("SVM,UB,", result)

    # SVM BB
    text_clf = Pipeline([
        ('vect',CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', SGDClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("SVM,BB,", result)

    # -----

    # RF UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', RandomForestClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("RF,UB,", result)

    # RF BB
    text_clf = Pipeline([
        ('vect',CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', RandomForestClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_evaluation.data)
    result = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    print("RF,BB,", result)

    # -----