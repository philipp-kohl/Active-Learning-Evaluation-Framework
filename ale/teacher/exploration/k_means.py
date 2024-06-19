import logging
from threading import Lock
from typing import List, Dict, Any

from numpy.linalg import norm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.base_trainer import Predictor

logger = logging.getLogger(__name__)
lock = Lock()

class ClusterDocument:
    def __init__(self, idx: int, cluster_idx: int, distance: float):
        self.idx = idx
        self.cluster_idx = cluster_idx
        self.distance = distance

class ClusteredDocuments:
    def __init__(self, documents: List[ClusterDocument], num_clusters: int):
        self.clusters = np.arange(0,num_clusters)
        self.documents = documents

    def get_clustered_docs_by_idx(self, indices: List[int]) -> List[ClusterDocument]:
        output = [doc for doc in self.documents if doc.idx in indices]
        return output


def tfidf_vectorize(id2text: Dict[int, str]):
    logger.info("Initial TFIDF-Vectorizing started.")
    vectorizer = TfidfVectorizer(analyzer="word", lowercase=True)
    X = vectorizer.fit_transform(id2text.values())
    logger.info("Initial TFIDF-Vectorizing finished.")
    return X


def cluster_documents(corpus: Corpus, k: int) -> ClusteredDocuments:
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
        clustered_documents: List[ClusteredDocuments] = []
        for i in range(len(ids)):
            idx = ids[i]
            tfidf_vector = npm_tfidf[i]
            distances = [norm(center - tfidf_vector) for center in centers]
            clustered_documents.append(ClusterDocument(idx,np.argmin(distances),np.min(distances)))
        clustered_obj = ClusteredDocuments(clustered_documents,len(centers))
        logger.info("Initial k-means clustering done.")
    finally:
        lock.release()

    return clustered_obj


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
        self.clustered_documents: ClusteredDocuments = cluster_documents(corpus=corpus, k=self.k)

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
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
        self.k = len(self.labels)
        self.clustered_documents = cluster_documents(corpus=corpus, k=self.k)

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        docs = self.clustered_documents.get_clustered_docs_by_idx(potential_ids)
        clusters = self.clustered_documents.clusters
        docs_per_cluster = int(step_size/len(clusters)) # equal distribution
        output_ids = []
        empty_clusters = []

        for cluster in clusters:
            potential_docs_cluster = [doc for doc in docs if doc.cluster_idx==cluster]
            sorted_docs_cluster = potential_docs_cluster.sort(key=lambda x: x.distance, reverse=True)
            if len(sorted_docs_cluster) < docs_per_cluster: # less docs in cluster left than needed
                output_ids.extend(sorted_docs_cluster)
                empty_clusters.append(cluster)
            else:
                output_ids.extend(sorted_docs_cluster[:docs_per_cluster])

        while step_size>len(output_ids) and len(empty_clusters)<len(clusters): # rest left
            docs_per_rest_clusters = max(int(step_size-len(output_ids))/(len(clusters)-len(empty_clusters)),1) # equally distribute to not empty clusters
            for cluster in clusters:
                if cluster not in empty_clusters:
                    potential_docs_cluster = [doc for doc in docs if doc.cluster_idx==cluster]
                    sorted_docs_cluster = potential_docs_cluster.sort(key=lambda x: x.distance, reverse=True)
                    if len(sorted_docs_cluster) < docs_per_rest_clusters: # less docs in cluster left than needed
                        rest -= len(sorted_docs_cluster)
                        output_ids.extend(sorted_docs_cluster)
                        empty_clusters.append(cluster)
                    else:
                        output_ids.extend(sorted_docs_cluster[:docs_per_rest_clusters])
        
        return output_ids