import logging
from collections import defaultdict
from typing import List
from typing import Optional, Dict, Tuple

import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from ale.corpus.corpus import Corpus
from ale.teacher.exploration.utils.silhouette_helper import silhouette_analysis

logger = logging.getLogger(__name__)


class ClusterDocument:
    def __init__(self, idx: int, cluster_idx: int, distance_to_center: float):
        self.idx = idx
        self.cluster_idx = cluster_idx
        self.distance = distance_to_center


class ClusteredDocuments:
    def __init__(self, documents: List[ClusterDocument], num_clusters: int):
        self.clusters: List[int] = list(np.arange(0, num_clusters))
        self.documents = documents
        self.id2doc: Dict[int, ClusterDocument] = {}
        self.cluster2doc: Dict[int, List[ClusterDocument]] = defaultdict(list)

        for doc in documents:
            self.id2doc[doc.idx] = doc
            self.cluster2doc[doc.cluster_idx].append(doc)

    def get_clustered_docs_by_idx(self, indices: List[int]) -> List[ClusterDocument]:
        return [self.id2doc[doc_id] for doc_id in indices]

    def get_docs_by_cluster_idx(self, cluster_idx: int) -> List[ClusterDocument]:
        return self.cluster2doc[cluster_idx]

    def get_potential_docs_by_cluster_idx(self, cluster_idx: int, potential_ids: List[int]) -> List[ClusterDocument]:
        return [d for d in self.cluster2doc[cluster_idx] if d.idx in potential_ids]


class ClusterHelper:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.clustered_docs: Optional[ClusteredDocuments] = None

    def adaptive_cluster(self, corpus: Corpus,
                         num_labels: int,
                         seed: int,
                         k_means_init: str = "k-means++") -> ClusteredDocuments:
        embeddings = normalize(self.embeddings, norm="l2")
        best_k: int = silhouette_analysis(num_labels, seed, "cosine", embeddings, k_means_init)

        logger.info(f"Initial k-means clustering with k={best_k} started.")
        # bert vectorize the dataset and apply k-means
        model = KMeans(n_clusters=best_k, init=k_means_init,
                       max_iter=300, n_init='auto', random_state=seed)
        model.fit(embeddings)

        # get the distance to the corresponding cluster centroid for each document
        centers = model.cluster_centers_
        clustered_documents: List[ClusterDocument] = []
        data = corpus.get_all_texts_with_ids()

        for i, doc_id in enumerate(list(data.keys())):
            distances = [norm(center - embeddings[i]) for center in centers]
            clustered_documents.append(ClusterDocument(doc_id, np.argmin(distances), np.min(distances)))
        self.clustered_docs = ClusteredDocuments(clustered_documents, model.n_clusters)

        logger.info("Initial k-means clustering done.")

        return self.clustered_docs

    def propose_nearest_neighbors_to_centroids(self,
                                               potential_ids: List[int],
                                               step_size: int,
                                               budget: int) -> List[int]:
        """
        Selects docs that are nearest to cluster centroids.
        """
        clusters = self.clustered_docs.clusters
        docs_per_cluster = int(step_size / len(clusters))  # equal distribution

        output_docs, empty_clusters = self.sample_from_clusters(clusters, docs_per_cluster, potential_ids)

        if empty_clusters:
            number_remaining_docs = step_size - len(output_docs)
            already_selected_docs = [d.idx for d in output_docs]
            remaining_potential_ids = [idx for idx in potential_ids if idx not in already_selected_docs]
            non_empty_clusters = [c for c in clusters if c not in empty_clusters]
            output_docs_for_remaining_docs = self.sample_remaining_docs_evenly_from_not_empty_clusters(
                number_remaining_docs,
                remaining_potential_ids,
                non_empty_clusters)
            output_docs.extend(output_docs_for_remaining_docs)

        out_ids = [item.idx for item in output_docs]
        return out_ids

    def sample_from_clusters(self, clusters, docs_per_cluster, potential_ids) \
            -> Tuple[List[ClusterDocument], List[int]]:
        output_docs: List[ClusterDocument] = []
        empty_clusters: List[int] = []

        for cluster in clusters:
            potential_docs_cluster = self.clustered_docs.get_potential_docs_by_cluster_idx(cluster, potential_ids)
            potential_docs_cluster.sort(key=lambda x: x.distance, reverse=True)
            if len(potential_docs_cluster) >= docs_per_cluster:
                output_docs.extend(potential_docs_cluster[:docs_per_cluster])
            else:  # fewer docs in cluster left than needed
                output_docs.extend(potential_docs_cluster)
                empty_clusters.append(cluster)

        return output_docs, empty_clusters

    def sample_remaining_docs_evenly_from_not_empty_clusters(self,
                                                             number_remaining_docs: int,
                                                             remaining_potential_ids: List[int],
                                                             non_empty_clusters: List[int]) -> List[ClusterDocument]:
        logger.info(f"Cluster {number_remaining_docs} from {non_empty_clusters} non-empty clusters.")
        remaining_output_docs: List[ClusterDocument] = []

        robin_counter = 0
        while len(remaining_output_docs) < number_remaining_docs:
            cluster_idx = robin_counter % len(non_empty_clusters)
            selected_cluster = non_empty_clusters[cluster_idx]

            possible_docs = self.clustered_docs.get_potential_docs_by_cluster_idx(selected_cluster,
                                                                                  remaining_potential_ids)
            if possible_docs:
                possible_docs.sort(key=lambda x: x.distance, reverse=True)
                remaining_output_docs.append(possible_docs[0])
                remaining_potential_ids.remove(possible_docs[0].idx)

            robin_counter += 1

        return remaining_output_docs
