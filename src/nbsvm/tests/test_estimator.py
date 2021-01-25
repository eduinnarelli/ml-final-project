from nbsvm import NBSVMEstimator

from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris


def test_nbsvm_check_estimator():
    """Check that NBSVMEstimator is compatible with Scikit-Learn API."""
    check_estimator(NBSVMEstimator)


def test_fit_NBSVMEstimator_dummy_data():
    """Assert that NBSVMEstimator methods are working using a toy dataset."""
    X, y = load_iris(return_X_y=True)
    clf = NBSVMEstimator(max_iter=100).fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (len(X), 3), "Probs should have correct shape."
    assert clf.score(X, y) > 0.9, "Model should have converged to train data."
