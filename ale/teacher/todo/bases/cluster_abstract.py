from abc import ABC
import random
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from ale.teacher.bases.base_teacher import BaseTeacher
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class ClusterBaseTeacher(BaseTeacher, ABC):
    """
    abstract cluster class, child of abstract teacher class
    """

    def __init__(self, step_size: int, needs_model: bool, budget: int, data: Path):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        self.run_cnt = 1
        self.df = pd.read_json(self.data_path, lines=True)

    def tfidf_vectorize(self, potential_ids: List[int]):
        df_out = self.df[self.df.id.isin(potential_ids)]
        vectorizer = TfidfVectorizer(analyzer="word")
        X = vectorizer.fit_transform(df_out.text)
        return X

    def doc2vec_vectorize(self, potential_ids: List[int]):
        """
        vectorizer for document similarity.
        Uses gensim's doc2vec
        """
        df_out = self.df[self.df.id.isin(potential_ids)]
        documents = [TaggedDocument(row.text, [i]) for i, row in df_out.iterrows()]
        model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
        return [model.dv[i] for i, _ in df_out.iterrows()]

    def bert_vectorize(self, potential_ids: List[int]):
        """
        vectorizer for document similarity.
        """
        # df_out = self.df[self.df.id.isin(potential_ids)]
        # bert_sentence_emb = nlu.load('embed_sentence.bert')
        # X = [np.array(bert_sentence_emb.predict(text).sentence_embedding_bert.values[0]) for text in df_out]
        # return X
        return None


class KMeansBase(ClusterBaseTeacher, ABC):
    def __init__(self, step_size: int, needs_model: bool, budget: int, data: Path, k=2):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        self.cluster = None
        self.k = k
        self.model = KMeans(n_clusters=self.k, init="k-means++", max_iter=300, n_init=5)

    def get_propose_ids(self, choice: int, potential_ids: List[int], size):
        cluster = self.df.iloc[self.cluster[choice]]
        potential_cluster_ids = list(cluster[cluster.id.isin(potential_ids)].id)
        rem_size = min(size, len(potential_cluster_ids))
        propose_ids = random.sample(potential_cluster_ids, k=rem_size)
        return propose_ids

    def propose(self, potential_ids: List[int]) -> List[int]:
        propose_ids = list()
        unseen = list(range(self.k))
        step_size = min(len(potential_ids), self.step_size)

        while len(propose_ids) < step_size:
            choice = random.choice(unseen)
            size = step_size - len(propose_ids)
            propose_ids += self.get_propose_ids(
                choice=choice, potential_ids=potential_ids, size=size
            )
            unseen.remove(choice)

        return propose_ids
