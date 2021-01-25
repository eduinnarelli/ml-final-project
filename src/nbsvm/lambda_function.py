import json
import logging
import os

import numpy as np

from nbsvm.saving import load_model

logger = logging.getLogger()
# Get log level from env variable. Defaults to INFO
loglevel = os.environ.get('LOG_LEVEL', 'INFO')
logger.setLevel(getattr(logging, loglevel))

# path for local debugging
# MODEL_ARTIFACT_PATH = '../models/baseline_classifier/model.pkl'
# Path (inside lambda zip pack) where the artifact is located.
MODEL_ARTIFACT_PATH = 'nbsvm/model.pkl'
NBSVM_MODEL = load_model(MODEL_ARTIFACT_PATH)


def predict(text):
    """ Uses NBSVM to classify a text segment

    Args:
        text (str): segment to be classified

    Returns:
        dict: with keys'predicted_class' and 'confidence' of segment.
    """
    probs = NBSVM_MODEL.predict([text])
    confidence = np.max(probs[0, :])
    class_idx = np.argmax(probs[0, :])
    label = NBSVM_MODEL.labels_names[int(class_idx)]

    prediction = {'predicted_class': label,
                  'confidence': confidence}

    return prediction


def lambda_handler(event, context):
    """ Receives an event containing text, runs the classifier and returns the
    results as a JSON.

    When invoking this function directly, `event` (the input) should be a JSON
    with a `text` field."""

    logger.debug("Received event: " + json.dumps(event, indent=2))

    text = event.get('text', '')
    logger.debug('Text: %s', text)

    if not os.path.isfile(MODEL_ARTIFACT_PATH):
        raise Exception('Model file not found.')

    response = {
        'success': False,
        'predicted_class': None,
        'confidence': 0.0,
    }

    if text:
        prediction = predict(text)

        response['success'] = True
        response['predicted_class'] = prediction['predicted_class']
        response['confidence'] = prediction['confidence']

    return response
