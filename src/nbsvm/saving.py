import joblib

from .nbsvm import NBSVM


def save_model(model_path, nbsvm_model):
    """ Saves a NBSVM model using joblib

    Args:
        model_path (str): Path to save the model
        nbsvm_model (nbsvm.NBSVM): model to be saved
    """
    with open(model_path, 'wb') as fd:
        joblib.dump(nbsvm_model, fd)


def load_model(model_path) -> NBSVM:
    """ Loads a NBSVM model using joblib

    Args:
        model_path (str): Path to load the model

    Returns:
        A NBSVM object.
    """
    with open(model_path, 'rb') as fd:
        model = joblib.load(fd)

    return model
