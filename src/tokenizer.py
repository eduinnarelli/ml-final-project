'''
Class to tokenize strings, i.e., to split texts into tokens before.
'''
import re
import string
from typing import List

from nltk.stem.snowball import SnowballStemmer


class Tokenizer:
    '''
    A Tokenizer object contains some tokenization hyperparameters as
    attributes.

    Attributes:
        do_stem: either to apply stemming on the tokens or not.
        min_word_size: token minimum length. If smaller than that, the token
            will be discarded.
    '''

    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def __init__(self, do_stem=True, min_word_size=4):
        self.do_stem = do_stem
        self.min_word_size = min_word_size
        self._stemmer = None

    def __repr__(self):
        return f'<Tokenizer ' \
            f'do_stem:{self.do_stem} ' \
            f'min_word_size:{self.min_word_size}>'

    def tokenize(self, text: str) -> List[str]:
        '''
        Splits text into tokens of at least `self.min_word_size` length.

        Args:
            text: text string.

        Returns:
            List of tokens in `text`.
        '''

        # Insert space before and after every punctuation char and then split
        tokens = self.re_tok.sub(r' \1 ', text)
        tokens = tokens.split()

        # Performs stemming (reduce inflected words to their stem) on tokens if
        # configured to.
        if self.do_stem:
            stemmer = SnowballStemmer('portuguese', ignore_stopwords=True)
            tokens = map(stemmer.stem, tokens)

        return [tok for tok in tokens if len(tok) >= self.min_word_size]
        
