import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from ale.corpus.corpus import Corpus

logger = logging.getLogger(__name__)


def is_named_entity(label: str) -> bool:
    """ Checks whether the given label is a named entity (not '0').
    """
    if label not in [None, "", "0", "O"]:
        return True
    return False


def tfidf_vectorize(texts: List[str], max_features: Optional[int] = None) -> np.ndarray:
    """ Vectorizes the given data with TFIDF

    Args:
        - texts (List[str]): A list of texts representing documents

    Returns:
        - X (spmatrix): An spmatrix containing the TFIDF embeddings for the corpus.
    """
    vectorizer = TfidfVectorizer(analyzer="word", lowercase=True, max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X


def sentence_transformer_vectorize(corpus: Corpus, model_name: str = "bert-base-nli-mean-tokens"):
    """ Vectorizes the given data with BERT

    Args:
        - corpus (Corpus): The corpus.

    Returns:
        - sentence_embeddings (ndarray): ND-Array containing the BERT embeddings for the corpus.
    """
    data: Dict[int, str] = corpus.get_all_texts_with_ids()
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(list(data.values()))
    return sentence_embeddings
