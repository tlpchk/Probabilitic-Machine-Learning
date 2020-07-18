from typing import List, Tuple

import gensim
import nltk
import torch
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups

from .common import MAX_DOCS_NUM, MAX_VOCAB_SIZE

nltk.download("wordnet")
stemmer = SnowballStemmer("english")


def stem_and_lemmatize(text: str) -> str:
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))


def preprocess(text: str) -> List[str]:
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(stem_and_lemmatize(token))

    return result


def get_newsgroup_dataset() -> Tuple[
    Tuple[List[List[str]], List[List[str]]], gensim.corpora.Dictionary
]:
    newsgroups_train = fetch_20newsgroups(subset="train", shuffle=True)
    newsgroups_test = fetch_20newsgroups(subset="test", shuffle=True)

    preprocessed_train = [preprocess(doc) for doc in newsgroups_train.data]
    preprocessed_test = [preprocess(doc) for doc in newsgroups_test.data]

    dictionary = gensim.corpora.Dictionary(preprocessed_train)
    dictionary.filter_extremes(
        no_below=15, no_above=0.1, keep_n=MAX_VOCAB_SIZE
    )
    dictionary.num_docs = min(MAX_DOCS_NUM, dictionary.num_docs)

    new_train = []
    for doc in preprocessed_train:
        train_sample = [word for word in doc if word in dictionary.token2id]
        if len(train_sample) > 0:
            new_train.append(train_sample)
    preprocessed_train = new_train

    new_test = []
    for doc in preprocessed_test:
        test_sample = [word for word in doc if word in dictionary.token2id]
        if len(test_sample) > 0:
            new_test.append(test_sample)
    preprocessed_test = new_test

    if MAX_DOCS_NUM > 0:
        preprocessed_train = preprocessed_train[:MAX_DOCS_NUM]
        preprocessed_test = preprocessed_test[:MAX_DOCS_NUM]

    return (preprocessed_train, preprocessed_test), dictionary
