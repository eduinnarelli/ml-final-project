'''
Function that builds the vectorization + training pipeline.
'''
from typing import Any, Dict

from nbsvm import NBSVMEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from tokenizer import Tokenizer


def build_pipeline(hparams: Dict[str, Any]) -> Pipeline:
    '''
    Builds a pipeline that vectorizes the data and trains the NBSVM model.

    Args:
        hparams: vectorizer and estimator hyperparameters.

    Returns:
        Sklearn pipeline.
    '''

    # Instantiate tokenizer
    tokenizer = Tokenizer(
        do_stem=hparams['do_stem'], min_word_size=hparams['min_word_size']
    )

    # The vectorizer builds a document-term matrix, appropriate to be fitted
    # by the model.
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',        # remove accents and apply char.
                                        # normalization;
        lowercase=hparams['do_lower'],  # either to convert chars. to lowercase
                                        # or not;
        tokenizer=tokenizer.tokenize,   # tokenize pre-processed texts;
        ngram_range=(1, 2),             # features will be unigrams (words) or
                                        # bigrams;
        min_df=hparams['min_df'],       # ignore lower frequencies;
        max_df=hparams['max_df'],       # ignore higher frequencies;
        sublinear_tf=True,              # apply sublinear scaling.
    )

    # The NBSVM estimator trains the model, i.e., tries to find the parameters
    # that better fit the vectorized data.
    nbsvm_estimator = NBSVMEstimator(
        C=hparams['C'],                 # inverse of regularization strength;
        dual=hparams['dual'],           # either to solve dual problem or not
                                        # (dual is better if n_samples <
                                        # n_features);
        solver=hparams['solver'],       # algorithm to use in optimization;
        max_iter=hparams['max_iter'],   # maximum number of iterations to
                                        # converge.
    )

    return Pipeline([('vectorizer', vectorizer), ('nbsvm', nbsvm_estimator)])
    
