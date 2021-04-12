import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from MBC_exploration import clf_pipe, check_performance, tokenize_and_lemma

if __name__ == '__main__':
    # (sys.argv[0]) # UB_BB.py
    trainset = (sys.argv[1])
    evalset = (sys.argv[2])
    output = (sys.argv[3])

    categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']

    train_folder = ".\Selected 20NewsGroup\\" + trainset
    evaluation_folder = ".\Selected 20NewsGroup\\" + evalset

    twenty_train = load_files(train_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
    twenty_evaluation = load_files(evaluation_folder, categories=categories, shuffle=True, random_state=42, encoding='latin1')
    docs_test = twenty_evaluation.data

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

    f = open(output, "w")

    # best configuration (lemma)
    vect = TfidfVectorizer(lowercase=True, tokenizer=tokenize_and_lemma, max_df=0.5, ngram_range=(1,3))
    clf = MultinomialNB(alpha=0.001)
    text_clf = clf_pipe(twenty_train, vect, clf)
    nb_c1 = (text_clf, check_performance(twenty_evaluation, text_clf))

    print(nb_c1)
    c1 = nb_c1[1]
    f.write("NB" + ",LB," + c1)
    f.close()