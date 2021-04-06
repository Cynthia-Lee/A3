# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?fbclid=IwAR2e_uoplSxSBWON3XZ69JA1Fnck-SFFE42PUKAVPi_quhe8CQk4qUnReWQ

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
nltk.download('stopwords')
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

### Tutorial: Loading the 20 newsgroups dataset
categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']
train_folder = ".\Selected 20NewsGroup\Training"

twenty_train = load_files(train_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
# {
# 'DESCR': None,
# 'data': [],
# 'filenames': array(),
# 'target': array(),
# 'target_names': []
# }

# print(twenty_train.target_names)
# print(len(twenty_train.data))
# print(len(twenty_train.filenames))
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print("\n".join(twenty_train.data[0].split("\n")[:4])) # header
# print("\n".join(twenty_train.data[0].split("\n")[4:])) # body
# print(twenty_train.target_names[twenty_train.target[0]]) # category

# -----------------------------------------------------------------------------------------

### Tutorial: Extracting features from text files
# vectorizer

# unigram baseline (UB)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data) # matrix
print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'algorithm')) # counts of N-grams of words or consecutive characters

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

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