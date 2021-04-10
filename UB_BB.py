# Basic Comparison with Baselines - For all four methods (NB,LR, SVM, and RF), 
# you should run both unigram and bigram baselines. (UB, BB)

# Hyperparameters
# 1. Naive Bayes - no hyperparameters 
# 2. Logistic Regression - Regularization constant, num iterations
# 3. SVM - Regularization constant, Linear, polynomial or RBF kernels.
# 4. RandomForest - Number of trees and number of features to consider.

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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

### Main class ###
if __name__ == '__main__':
    # (sys.argv[0]) # UB_BB.py
    trainset = (sys.argv[1])
    evalset = (sys.argv[2])
    output = (sys.argv[3])
    display_LC = (sys.argv[4]) # `1' to show the plot

    categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']

    train_folder = ".\Selected 20NewsGroup\\" + trainset
    evaluation_folder = ".\Selected 20NewsGroup\\" + evalset

    twenty_train = load_files(train_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
    twenty_evaluation = load_files(evaluation_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
    docs_test = twenty_evaluation.data

    f = open(output, "w")

    # -----
    
    # NB UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', MultinomialNB()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    # print(np.mean(predicted == twenty_evaluation.target)) # accuracy
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    # precision, # recall, fbeta_score, support
    result = ",".join(map(str,info[:-1]))
    f.write("NB,UB," + result + "\n")
    
    # NB BB
    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', MultinomialNB()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("NB,BB," + result + "\n")

    # -----

    # LR UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', LogisticRegression()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("LR,UB," + result + "\n")

    # LR BB
    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', LogisticRegression()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("LR,BB," + result + "\n")

    # -----

    # SVM UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', SGDClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("SVM,UB," + result + "\n")

    # SVM BB
    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', SGDClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("SVM,BB," + result + "\n")

    # -----

    # RF UB
    text_clf = Pipeline([
        ('vect', CountVectorizer()), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', RandomForestClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("RF,UB," + result + "\n")

    # RF BB
    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer='word', ngram_range=(2, 2))), # vector
        ('tfidf', TfidfTransformer()), # transformer
        ('clf', RandomForestClassifier()), # classifier
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    info = precision_recall_fscore_support(twenty_evaluation.target, predicted, average='macro')
    result = ",".join(map(str,info[:-1]))
    f.write("RF,BB," + result)

    f.close()

    # -----

    # learning curve (LC)
    # show the performance of each classifier only with the unigram representation.
    # The learning curve is a plot of the performance of the classifier 
    # (F1-score on the y-axis) on the evaluation data, when trained on different amounts of training data (size of training data on the x-axis).

    def learning_curve(train_data, train_target, t_size, classifier):
        text_clf = Pipeline([
            ('vect', CountVectorizer()), # vector
            ('tfidf', TfidfTransformer()), # transformer
            ('clf', classifier), # classifier
        ])
        if (t_size == 1.0):
            text_clf.fit(twenty_train.data, twenty_train.target) # text_clf.fit(train_data, train_target)
        else:
            X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, train_size=t_size)
            text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(docs_test) # twenty_evaluation.data
        score = f1_score(twenty_evaluation.target, predicted, average='macro')
        return score

    # You are given the validation fold, ie, evaluation data, and do not need to repeat many times. 
    # All you need is to train on the training data with varying sizes and get F1 score on the evaluation data.

    # https://scikit-learn.org/stable/modules/cross_validation.html

    if (display_LC == '1'):
        training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        nb_f1_arr = []
        lr_f1_arr = []
        svm_f1_arr = []
        rf_f1_arr = []
        for t_size in training_sizes:
            nb = learning_curve(twenty_train.data, twenty_train.target, t_size, MultinomialNB())
            nb_f1_arr.append(nb)
            lr = learning_curve(twenty_train.data, twenty_train.target, t_size, LogisticRegression())
            lr_f1_arr.append(lr)
            svm = learning_curve(twenty_train.data, twenty_train.target, t_size, SGDClassifier())
            svm_f1_arr.append(svm)
            rf = learning_curve(twenty_train.data, twenty_train.target, t_size, RandomForestClassifier())
            rf_f1_arr.append(rf)

        print("NB", nb_f1_arr)
        print("LR", lr_f1_arr)
        print("SVM", svm_f1_arr)
        print("RF", rf_f1_arr)

        x_label = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
        plt.xlabel('Size of training data (%)')
        plt.ylabel('F1-score')
        plt.ylim(0.0, 1.0)

        plt.plot(x_label, nb_f1_arr, '-^', color='red') #NB
        plt.plot(x_label, lr_f1_arr, '-x', color='violet') # LR
        plt.plot(x_label, svm_f1_arr, '->', color='forestgreen') # SVM
        plt.plot(x_label, rf_f1_arr, '-o', color='orange') # RF
        plt.show()