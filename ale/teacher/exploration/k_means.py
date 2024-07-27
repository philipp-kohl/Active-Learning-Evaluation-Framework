import logging
from typing import List, Any

import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans

from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.teacher.exploration.utils.cluster_helper import ClusterHelper, ClusteredDocuments, ClusterDocument
from ale.teacher.exploration.utils.silhouette_helper import silhouette_analysis
from ale.teacher.teacher_utils import tfidf_vectorize, sentence_transformer_vectorize
from ale.trainer.predictor import Predictor

logger = logging.getLogger(__name__)


def cluster_documents(corpus: Corpus, num_labels: int, seed: int) -> ClusteredDocuments:
    data = corpus.get_all_texts_with_ids()
    ids = list(data.keys())
    X = tfidf_vectorize(texts=list(data.values()))

    best_k: int = silhouette_analysis(num_labels, seed, "euclidian", X)

    logger.info(f"Initial k-means clustering with k={best_k} started.")
    # tfidf vectorize the dataset and apply k-means++
    model = KMeans(n_clusters=best_k, init='k-means++',
                   max_iter=300, n_init='auto')
    model.fit(X)

    # get the distance to the corresponding cluster centroid for each document
    npm_tfidf = X.todense()
    centers = model.cluster_centers_
    clustered_documents: List[ClusterDocument] = []
    for i in range(len(ids)):
        idx = ids[i]
        tfidf_vector = npm_tfidf[i]
        distances = [norm(center - tfidf_vector) for center in centers]
        clustered_documents.append(ClusterDocument(idx, np.argmin(distances), np.min(distances)))
    clustered_docs = ClusteredDocuments(clustered_documents, len(centers))
    logger.info("Initial k-means clustering done.")

    return clustered_docs



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
        self.clustered_documents: ClusteredDocuments = cluster_documents(corpus=corpus, num_labels=self.k, seed=seed)

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:
        docs = self.clustered_documents.get_clustered_docs_by_idx(potential_ids)
        sorted_docs = docs.sort(key=lambda x: x.distance, reverse=True)
        out_ids = [item.idx for item in sorted_docs[:step_size]]

        return out_ids


@TeacherRegistry.register("k-means-cluster-based")
class KMeansClusterBasedTeacher(BaseTeacher):
    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task
        )
        self.num_labels = len(self.labels)
        self.clustered_documents = cluster_documents(corpus=corpus, num_labels=self.num_labels, seed=seed)

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:
        pass#return propose_nearest_neighbors_to_centroids(self.clustered_documents, potential_ids, step_size, budget)


@TeacherRegistry.register("k-means-cluster-based-bert-km")
class KMeansClusterBasedBERTTeacher(BaseTeacher):
    """
    M. Van Nguyen, N. T. Ngo, B. Min, and T. H. Nguyen,
    “FAMIE: A Fast Active Learning Framework for Multilingual Information Extraction” presented at the NAACL 2022 -
    2022 Conference of the North American Chapter of the Association for Computational Linguistics:
    Human Language Technologies, Proceedings of the Demonstrations Session, 2022, pp. 131–139.

    Notes:
      - Paper cites: https://aclanthology.org/2020.emnlp-main.637.pdf
        Information taken: use k-means instead of k-means++, use [CLS] token, L2 normalized embeddings

    Questions: which bert model? How to define k? They used 100 and 200?
    We take SOTA sentence transformer and define k based on silhouette_analysis.

    """

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task
        )
        self.num_labels = len(self.labels)
        # all-mpnet-base-v2: currently (07-2024) the SOTA sentence transformer
        embeddings = sentence_transformer_vectorize(corpus, model_name="all-mpnet-base-v2")
        self.cluster_helper = ClusterHelper(embeddings)
        self.cluster_helper.adaptive_cluster(corpus=corpus,
                                             num_labels=self.num_labels,
                                             seed=seed)

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:
        return self.cluster_helper.propose_nearest_neighbors_to_centroids(potential_ids, step_size, budget)
