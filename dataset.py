# Header
# Consists of fields such as <From>, <Subject>, <Organization> and <Lines> fields.
# The <lines> field includes the number of lines in the document body
# Body
# The main body of the document. This is where you should extract features from.

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?fbclid=IwAR2e_uoplSxSBWON3XZ69JA1Fnck-SFFE42PUKAVPi_quhe8CQk4qUnReWQ

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

### Loading the 20 newsgroups dataset
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

### Extracting features from text files
# feature representations
# feature extractors first segment the text into sequences and tokenize them into words.
# each document is then represented as a vector based on the words that occur in it.
# 1. CountVectorizer (bag of words) - Uses the number of times each word was observed.
# 2 TFIDFVectorizer - Uses relative frequencies normalized by the inverse of the number of documents in which the word was observed

count_vect = CountVectorizer() # builds dictionary of features and transofmrs documents to feature vectors
X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'algorithm')) # counts of N-grams of words or consecutive characters