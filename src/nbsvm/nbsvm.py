import re
import string

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DEFAULT_TOKENS = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s):
    return DEFAULT_TOKENS.sub(r' \1 ', s).split()


def _pr(y_i, y, x):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def _fit_model(y, x):
    try:
        y = y.values
    except AttributeError:
        pass
    r = np.log(_pr(1, y, x) / _pr(0, y, x))
    model = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return model.fit(x_nb, y), r


class NBSVM:
    def __init__(self, labels, tokenizer=tokenize, **kwargs):
        """ Instantiates a NBSVM model.

        Pass extra argments to control how the internal TfidfVectorizer is
        created. Splitter_tokens defines the tokenizer to be used. Example:

        >>> model = NBSVM(labels=['label_1', 'label_2'],
        >>>               ngram_range=(1, 2), min_df=3, max_df=0.9,
        >>>               strip_accents='unicode', use_idf=1, smooth_idf=1,
        >>>               sublinear_tf=1)

        Args:
            labels (list[str]): unique names for each label
            tokenizer (callable): a function to be used as argument to
                `TfidfVectorizer` class.
            **kwargs: argments that will be used to create a
                sklearn.feature_extraction.text.TfidfVectorizer object.
        """
        self._models = dict()

        for label in labels:
            self._models[label] = tuple()
        self._vectorizer = TfidfVectorizer(tokenizer=tokenizer, **kwargs)

    @property
    def vectorizer(self):
        return self._vectorizer

    @property
    def models(self):
        return self._models

    @property
    def labels_names(self):
        return list(self._models.keys())

    def fit(self, onehot_labels_dict, texts):
        """ Fits the model according to a training set of texts.

        Args:
            onehot_labels_dict (dict[str, np.ndarray]): Dictionary
                where each key corresponds to a label. Each value is an 1d-array
                where each i-th element is binary and it's 1 if the i-th text is
                of that label.
            texts (list or list-like): iterable where each element is a text
                to be trained on.
        """
        texts = self.vectorizer.fit_transform(texts)
        for i, label in enumerate(self._models):
            print('\nfit', label)
            model, ratio = _fit_model(onehot_labels_dict[label],
                                      texts)
            self._models[label] = (model, ratio)

    def predict(self, texts) -> np.ndarray:
        """ Predicts a list or list-like of texts.

        Args:
            texts (list or list-like): iterable where each element is a text to
                predict.

        Returns:
            Predictions in the form of a 2d-matrix.

        Shapes:
            - Input: `:math: (N)`
            - Output: `:math: (N, M)` where M is the number of labels.
        """
        labels = self._models.keys()
        preds = np.zeros((len(texts), len(labels)))
        texts = self.vectorizer.transform(texts)
        for i, label in enumerate(labels):
            model, ratio = self._models[label]
            preds[:, i] = model.predict_proba(texts.multiply(ratio))[:, 1]

        return preds
