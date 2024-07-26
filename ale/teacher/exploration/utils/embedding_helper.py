from typing import Dict, List, Callable

import numpy as np

from ale.corpus.corpus import Corpus


class EmbeddingHelper:
    def __init__(self, corpus: Corpus, embedding_function: Callable[[List[str]], np.ndarray]):
        self.doc_id2embedding_index: Dict[int, int] = {}
        texts: List[str] = []
        for i, (doc_id, text) in enumerate(corpus.get_all_texts_with_ids().items()):
            self.doc_id2embedding_index[doc_id] = i
            texts.append(text)

        self.embeddings = embedding_function(texts)

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings

    def get_embedding_index_for_doc_id(self, idx: int) -> int:
        if idx not in self.doc_id2embedding_index:
            raise ValueError("Given id not in corpus.")

        return self.doc_id2embedding_index[idx]

    def get_embedding_indices_for_doc_ids(self, ids: List[int]) -> List[int]:
        return [self.get_embedding_index_for_doc_id(idx) for idx in ids]
