from typing import Optional

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NBSVMEstimator(BaseEstimator):
    """NBSVM (Naive-Bayes SVM) estimator compatible with scikit-learn API.

    Given a multi-class classification problem with `K` classes, this estimator
    will fit `K` LogisticRegression models in a one-vs-rest scheme. Each
    class has a distinct `r` array that projects the input features before
    feeding to the corresponding classifier.

    This implementation follows Jeremy Howards' Kaggle kernel:
    https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

    NOTE: since this estimator uses one-vs-rest scheme, the class probabilities
    output of `predict_proba` method for a given example will **not** sum to
    one.

    After the model is trained with `fit`, the following attributes will be set:
        classes_: np.ndarray of shape (K,) with all `K` data labels.
        models_: a tuple with `K` LogisticRegresson instances, one per class, in
            the same order as they appear in `classes_` attribute.
        rs_: a tuple with `K` np.ndarray instances, one per class, in the same
            order as they appear in `classes_` attribute.


    Args:
        C: Inverse of regularization strength used in `LogisticRegression`. Must
            be a positive float. Smaller values specify stronger regularization.
        dual: `LogisticRegression` parameter. See Scikit-Learn docs for info.
        solver: `LogisticRegression` parameter. See Scikit-Learn docs for info.
        max_iter: Maximum number of iterations taken for the solvers to
            converge.
        n_jobs: `LogisticRegression` parameter, see Scikit-Learn docs for info.
            NOTE: this parameter is ignored when using `liblinear` (default)
            solver.
        random_state: the seed of the pseudo random generator. Passed directly
            to `LogisticRegression`, see Scikit-Learn docs for more info.
    """

    def __init__(self,
                 C: int = 4,
                 dual: bool = True,
                 solver: str = 'liblinear',
                 max_iter: int = 100,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 ) -> None:
        self.C = C
        self.dual = dual
        self.solver = solver
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _clear_state(self) -> None:
        """Clears the estimator state."""
        self.rs_ = []
        self.models_ = []
        self.classes_ = None

    @staticmethod
    def _fit_r(X, y, y_i) -> np.ndarray:
        """Fits the Naive-Bayes `r` projection array for a given class.

        Args:
            X: the array of features of shape (N, M).
            y: the array of labels of shape (N,).
            y_i: the class (int or string).

        Returns:
            The computed `r` vector of shape (M,).
        """
        if issparse(X):
            X = X.tocsr()

        ix_pos = (y == y_i)
        ix_neg = ~ix_pos

        p_pos = X[ix_pos].sum(0)
        pr_pos = (p_pos + 1) / (ix_pos.sum() + 1)

        p_neg = X[ix_neg].sum(0)
        pr_neg = (p_neg + 1) / (ix_neg.sum() + 1)

        return np.asarray(np.log(pr_pos / pr_neg)).squeeze()

    @staticmethod
    def _project_x(X, r) -> np.ndarray:
        """Projects the 2d feature array to same shape using Naive Bayes' `r` 1d
        array."""
        if isinstance(X, np.ndarray):
            return X * r

        # Sparse matrix or DataFrame
        return X.multiply(r)

    def fit(self, X, y) -> 'NBSVMEstimator':
        """Fits the estimator.

        Args:
            X: 2d array of features of shape (N, M), where `N` is the number of
                samples and `M` is the number of features.
            y: 1d array of labels of shape (N,) with K distinct classes in a
                multi-class style. The dtype can be integer or string.
                NOTE: there must be at least one example of each class.

        Returns:
            The trained estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        self._clear_state()
        self.classes_ = unique_labels(y)

        for label in self.classes_:
            # Get array of labels in a one-vs-rest scheme
            y_label = (y == label)

            model = LogisticRegression(
                C=self.C,
                solver=self.solver,
                dual=self.dual,
                n_jobs=self.n_jobs,
                max_iter=self.max_iter,
                random_state=self.random_state)

            r = self._fit_r(X, y, label)
            x_nb = self._project_x(X, r)
            model = model.fit(x_nb, y_label)

            self.models_.append(model)
            self.rs_.append(r)

        self.models_ = tuple(self.models_)
        self.rs_ = tuple(self.rs_)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predicts the class probabilities for input samples.

        Args:
            X: input 2d array of shape (N, M).

        Returns:
            2d array of probabilities of shape (N, K), where `K` is the number
            of classes inferred during `fit`.
        """
        check_is_fitted(self, ['classes_', 'rs_', 'models_'])
        X = check_array(X, accept_sparse=True)

        preds = np.empty((X.shape[0], len(self.models_)), dtype=np.float)

        for label_ix, (r, model) in enumerate(zip(self.rs_, self.models_)):
            x_nb = self._project_x(X, r)
            preds[:, label_ix] = model.predict_proba(x_nb)[:, 1]

        return preds

    def predict(self, X) -> np.ndarray:
        """Predicts the classes for input samples.

        Args:
            X: 2d input feature array of shape (N, M).

        Returns:
            1d array of predicted classes of shape (N,). The dtype will be the
            same of `y` given in `fit` method.
        """
        probs = self.predict_proba(X)
        pred_ix = np.argmax(probs, axis=1)

        # Return predicted classes
        return self.classes_[pred_ix]

    def score(self, X, y) -> float:
        """Computes accuracy classification score."""
        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)
