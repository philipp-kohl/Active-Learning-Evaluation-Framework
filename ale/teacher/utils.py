from sklearn.feature_extraction.text import TfidfVectorizer
from ale.corpus.corpus import Corpus
from typing import Dict
from scipy.sparse import spmatrix


def is_named_entity(label: str) -> bool:
    """ Checks whether the given label is a named entity (not '0').
    """
    if label not in [None, "", "0", "O"]:
        return True
    return False


def tfidf_vectorize(id2text: Dict[int, str]):
    """ Vectorizes the given data with TFIDF

    Args:
        - id2text (Dict[int, str]): A dict with the document's ids as keys and the texts as values.

    Returns:
        - X (spmatrix): An spmatrix containing the TFIDF embeddings for the corpus.
    """
    vectorizer = TfidfVectorizer(analyzer="word", lowercase=True)
    X = vectorizer.fit_transform(id2text.values())
    return X


def embed_documents_with_tfidf(corpus: Corpus) -> spmatrix:
    """ Calculates embeddings for the given corpus data.

    Args:
        - corpus (Corpus): The corpus.
    
    Returns: 
        - X (spmatrix): An spmatrix containing the TFIDF embeddings for the corpus.
    """
    # tfidf vectorize the dataset
    data: Dict[int, str] = corpus.get_all_texts_with_ids()
    return tfidf_vectorize(id2text=data)