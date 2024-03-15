import logging

from typing import List, Dict, Any
from threading import Lock

from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.base_trainer import Predictor

logger = logging.getLogger(__name__)
lock = Lock()

def tfidf_vectorize(id2text: Dict[int, str]):
    logger.info("Initial TFIDF-Vectorizing started.")
    vectorizer = TfidfVectorizer(analyzer="word", lowercase=True)
    X = vectorizer.fit_transform(id2text.values())
    logger.info("Initial TFIDF-Vectorizing finished.")
    return X


def cluster_documents(corpus: Corpus, k: int):
    lock.acquire()
    try:
        logger.info(f"Initial k-means clustering with k={k} started.")
        # tfidf vectorize the dataset and apply k-means++
        model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init='auto')
        data = corpus.get_all_texts_with_ids()
        ids = list(data.keys())
        X = tfidf_vectorize(id2text=data)
        model.fit(X)

        # get the distance to the corresponding cluster centroid for each document
        npm_tfidf = X.todense()
        idx2distance = dict()
        centers = model.cluster_centers_
        for i in range(len(ids)):
            idx = ids[i]
            tfidf_vector = npm_tfidf[i]
            distances = [norm(center - tfidf_vector) for center in centers]
            idx2distance[idx] = min(distances)

        logger.info("Initial k-means clustering done.")
    finally:
        lock.release()

    return idx2distance


@TeacherRegistry.register("k-means")
class KMeansTeacher(BaseTeacher):

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task
        )
        self.k = len(self.labels)
        self.idx2distance = cluster_documents(corpus=corpus, k=self.k)

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        potential_distances = {idx: self.idx2distance[idx] for idx in potential_ids}
        sorted_dict_by_score = sorted(potential_distances.items(), key=lambda x:x[1], reverse=True)
        out_ids = [item[0] for item in sorted_dict_by_score[:step_size]]

        return out_ids
