from typing import List, Any, Dict, Optional
import random
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.teacher.exploitation.aggregation_methods import AggregationMethod
from ale.teacher.exploration.utils.cluster_helper import ClusteredDocuments, ClusterDocument, ClusterHelper
from ale.teacher.exploration.utils.silhouette_helper import silhouette_analysis
from ale.trainer.predictor import Predictor
from ale.trainer.prediction_result import TokenConfidence, PredictionResult


class NGramVectors:
    def __init__(self, sizes: List[int], seed: int, lexical_dimension: int = 50) -> None:
        self.sizes: List[int] = sizes
        self.seed: int = seed
        self.lexical_dimension: int = lexical_dimension
        self.vector_dict: Dict[str, np.ndarray] = dict()

    def get_lexical_token_vector(self, token: str) -> np.ndarray:
        """
        Get the normalized sum of the token n-grams
        """
        vectors: List[np.ndarray] = self.generate_lexical_token_vector(token)
        vector_sum: np.ndarray = np.sum(vectors, axis=0)
        norm = np.linalg.norm(vector_sum)
        if norm == 0:
            return vector_sum
        return vector_sum / norm

    def generate_lexical_token_vector(self, token: str) -> List[np.ndarray]:
        """
        Generates n grams of the given sizes
        """
        subsequences: List[str] = []
        vectors: List[np.ndarray] = []
        for size in self.sizes:
            last_start_index = len(token) - size + 1
            subsequences_of_size: List[str] = [token[i:i + size] for i in range(last_start_index)]
            subsequences.extend(subsequences_of_size)

        if subsequences:
            # given token is longer than sizes for subsequences
            for seq in subsequences:
                subsequence_vector: np.ndarray = self.get_lexical_subsequence_vector(seq)
                vectors.append(subsequence_vector)
        else:
            subsequence_vector: np.ndarray = self.get_lexical_subsequence_vector(token)
            vectors.append(subsequence_vector)
        return vectors

    def get_lexical_subsequence_vector(self, seq: str) -> np.ndarray:
        seq_vector = self.vector_dict.get(seq)
        if seq_vector is None:  # vector n gram does not exist yet
            np.random.seed(self.seed)  # use ALE seed
            # lexical vector of dimension given, per default 50
            self.vector_dict[seq] = np.random.random(size=self.lexical_dimension)
        return self.vector_dict[seq]


def embed_documents_with_lexical_and_semantic_vectors(corpus: Corpus, ngrams: NGramVectors,
                                                      word_embedding_dimension: int = 100) -> Dict[int, np.ndarray]:
    """
    Calculates lexical vectors with n-grams (bi- and tri-grams) and semantic vectors by Skip-grams (Word2Vec),
    concatenates and normalizes them
    """
    dict_tokens: Dict[int, List[str]] = corpus.get_all_tokens()
    tokenized_texts: List[List[str]] = list(dict_tokens.values())
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=word_embedding_dimension,
                              window=5, min_count=1, workers=4)  # semantic vectors

    vectors: Dict[int, np.ndarray] = {
        doc_id: embed_single_doc(tokens, ngrams, word2vec_model)
        for doc_id, tokens in dict_tokens.items()
    }

    return vectors


def embed_single_doc(tokens: List[str], ngrams: NGramVectors, word2vec_model: Word2Vec):
    doc_vectors: List[np.ndarray] = []
    lexical_vectors: List[np.ndarray] = [ngrams.get_lexical_token_vector(token) for token in tokens]
    semantic_vectors: List[np.ndarray] = [word2vec_model.wv[token] for token in tokens]

    for lex, sem in zip(lexical_vectors, semantic_vectors):
        # concatenate semantic and lexical vectors for each token
        combined: np.ndarray = np.concatenate((sem, lex))
        doc_vectors.append(combined)

    sum_vectors = np.add.reduce(doc_vectors)
    norm = np.linalg.norm(sum_vectors)
    if norm == 0:
        return sum_vectors
    else:
        # normalized sum of all token vectors is doc vector
        return sum_vectors / norm


def cluster_documents(num_labels: int, embeddings_dict: Dict[int, np.ndarray], seed) -> ClusteredDocuments:
    embeddings = list(embeddings_dict.values())
    best_k: int = silhouette_analysis(num_labels, seed, "cosine", embeddings)
    model = KMeans(n_clusters=best_k, init='k-means++',
                   max_iter=300, n_init='auto')
    model.fit(embeddings)

    # get the distance to the corresponding cluster centroid for each document
    centers = model.cluster_centers_
    clustered_documents: List[ClusterDocument] = []
    for idx in list(embeddings_dict.keys()):
        vector = embeddings_dict[idx]
        distances = [np.linalg.norm(center - vector) for center in centers]
        clustered_documents.append(ClusterDocument(idx, np.argmin(distances), np.min(distances)))
    clustered_obj = ClusteredDocuments(clustered_documents, len(centers))

    return clustered_obj


@TeacherRegistry.register("sequential-representation-least-confidence")
class SequentialRepresentationLCTeacher(BaseTeacher):
    """
    The sequential representative and least confidence teacher proposes samples that are most similar to
    unlabeled samples with the least confidence in the predictions

    Applied to ER task:
        - Kholghi, M., De Vine, L., Sitbon, L., Zuccon, G., Nguyen, A.:
        Clinical information extraction using small data: An active learning approach based on sequence
        representations and word embeddings. Journal of the Association for Information Science and Technology 68(11),
        2543–2556 (2017). doi: 10.1002/asi.23936 (in paper AL strategy is called "Unsupervised Least Confidence" (ULC))

    Notes:
        - Details in https://aclanthology.org/U15-1003.pdf

    """

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask,
                 aggregation_method: Optional[AggregationMethod]):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task,
            aggregation_method=aggregation_method
        )
        self.num_labels = len(self.labels)
        self.embeddings: Dict[int, np.ndarray] = embed_documents_with_lexical_and_semantic_vectors(
            corpus=corpus, ngrams=NGramVectors([2, 3], seed))
        self.clustered_documents: ClusteredDocuments = cluster_documents(self.num_labels, self.embeddings, seed)
        self.corpus = corpus

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:
        # get lc for documents (inside budget)
        batch = random.sample(potential_ids, budget)
        confidence_per_doc = self.make_predictions(batch)

        clustered_docs: Dict[int, List[int]] = self.get_lc_list_for_clusters(batch, confidence_per_doc)
        out_ids: List[int] = self.sample_round_robin(clustered_docs, step_size)

        return out_ids

    def sample_round_robin(self, clustered_docs, step_size):
        """
         Round-robin for clusters, always take doc with lowest lc
        """
        robin: int = 0  # Iterate over clusters
        num_clusters: int = len(self.clustered_documents.clusters)
        out_ids: List[int] = []

        while len(out_ids) < step_size:
            cluster_index = robin % num_clusters
            cluster = self.clustered_documents.clusters[cluster_index]
            # always add doc of cluster with max LC (lowest highest confidence)
            if clustered_docs[cluster]:
                out_ids.append(clustered_docs[cluster].pop(0))
            robin += 1
        return out_ids

    def get_lc_list_for_clusters(self, batch, confidence_per_doc) -> Dict[int, List[int]]:
        """
         Get sorted dict for all clusters and their best docs in descending order
         {cluster: [idx]} in descending order
        """
        clustered_docs: Dict[int, List[int]] = self.get_docs_in_clusters(batch)
        for key in clustered_docs.keys():
            docs = clustered_docs[key]
            lc_docs = {doc: confidence_per_doc[doc] for doc in docs}
            sorted_dict_by_lc = sorted(lc_docs.items(), key=lambda x: x[1])
            clustered_docs[key] = [key for key, value in sorted_dict_by_lc]
        return clustered_docs

    def make_predictions(self, batch: List[int]) -> Dict[int, float]:
        idx2text = self.corpus.get_text_by_ids(batch)
        prediction_results: Dict[int, PredictionResult] = self.predictor.predict(idx2text)
        confidence_per_doc: Dict[int, float] = self.compute_lc(prediction_results)
        return confidence_per_doc

    def get_docs_in_clusters(self, docs: List[int]) -> Dict[int, List[int]]:
        docs_by_cluster: Dict[int, List[int]] = {}
        for cluster in self.clustered_documents.clusters:
            doc_indices = [d.idx for d in self.clustered_documents.get_potential_docs_by_cluster_idx(cluster, docs)]
            docs_by_cluster[cluster] = doc_indices

        return docs_by_cluster

    def compute_lc(self, predictions: Dict[int, PredictionResult]) -> Dict[int, float]:
        """
        LC is calculated on token-level and aggregated on instance-level as configured.
        """
        scores: Dict[int, float] = {}
        for idx, prediction in predictions.items():
            token_confidences: List[TokenConfidence] = prediction.ner_confidences_token
            highest_confidences: List[float] = [token.get_highest_confidence().confidence
                                                for token in token_confidences]
            instance_score = self.aggregate_function(highest_confidences)
            scores[idx] = instance_score
        return scores

    def compute_ner(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, float]:
        """
          Not implemented
        """
        raise NotImplementedError("Not implemented so far")

    def compute_cls(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, int]:
        """
          Not implemented
        """
        raise NotImplementedError("Hybrid teacher is not implemented for text classification task.")
