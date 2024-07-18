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
from ale.trainer.predictor import Predictor
from ale.teacher.teacher_utils import ClusteredDocuments, ClusterDocument
from ale.trainer.prediction_result import TokenConfidence, PredictionResult
from ale.teacher.exploration.k_means import silhouette_analysis


class NGramVectors:
    def __init__(self, sizes: List[int], seed: int, lexical_dimension: int = 50) -> None:
        self.sizes: List[int] = sizes
        self.seed: int = seed
        self.lexical_dimension: int = lexical_dimension
        self.vector_dict: Dict[str, np.ndarray] = dict()

    def get_lexical_subsequence_vector(self, seq: str) -> np.ndarray:
        seq_vector = self.vector_dict.get(seq)
        if seq_vector is None:  # vector n gram does not exist yet
            np.random.seed(self.seed)  # use ALE seed
            # lexical vector of dimension given, per default 50
            self.vector_dict[seq] = np.random.random(
                size=(self.lexical_dimension))
        return self.vector_dict[seq]

    def generate_lexical_token_vector(self, token: str) -> List[np.ndarray]:
        """ Generates n grams of the given sizes
        """
        subsequences: List[str] = []
        vectors: List[np.ndarray] = []
        for size in self.sizes:
            subsequences_of_size: List[str] = [
                token[i:i+size] for i in range(len(token)-size)]
            subsequences.extend(subsequences_of_size)

        if len(subsequences) > 0:  # given token is longer than sizes for subsequences
            for seq in subsequences:
                subsequence_vector: np.ndarray = self.get_lexical_subsequence_vector(
                    seq)
                vectors.append(subsequence_vector)
        else:
            subsequence_vector: np.ndarray = self.get_lexical_subsequence_vector(
                token)
            vectors.append(subsequence_vector)
        return vectors

    def get_lexical_token_vector(self, token: str) -> np.ndarray:
        """ Get the normalized sum of the token n-grams
        """
        vectors: List[np.ndarray] = self.generate_lexical_token_vector(token)
        vector_sum: np.ndarray = np.sum(vectors, axis=0)
        norm = np.linalg.norm(vector_sum)
        if norm == 0:
            return vector_sum
        return vector_sum/norm


def embed_documents_with_lexical_and_semantical_vectors(corpus: Corpus, ngrams: NGramVectors, word_embedding_dimension: int = 100) -> Dict[int, np.ndarray]:
    """ Calculates lexical vectors with n-grams (bi- and tri-grams) and semantical vectors by Skip-grams (Word2Vec), concatenates and normalizes them
    """
    vectors: Dict[int, np.ndarray] = {}
    dict_tokens: Dict[int, List[str]] = corpus.get_all_tokens()
    tokenized_texts: List[List[str]] = list(dict_tokens.values())
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=word_embedding_dimension,
                              window=5, min_count=1, workers=4)  # semantic vectors
    for doc_id, tokens in dict_tokens.items():
        doc_vectors: List[np.ndarray] = []
        lexical_vectors: List[np.ndarray] = [
            ngrams.get_lexical_token_vector(token) for token in tokens]
        semantic_vectors: List[np.ndarray] = [
            word2vec_model.wv[token] for token in tokens]
        for lex, sem in zip(lexical_vectors, semantic_vectors):
            # concatenate semantic and lexical vectors for each token
            combined: np.ndarray = np.concatenate((sem, lex))
            doc_vectors.append(combined)
        sum_vectors = np.add.reduce(doc_vectors)
        norm = np.linalg.norm(sum_vectors)
        if norm == 0:
            vectors[doc_id] = sum_vectors
        else:
            # normalized sum of all token vectors is doc vector
            vectors[doc_id] = sum_vectors/norm
    return vectors


def cluster_documents(nr_labels: int, embeddings: Dict[int, np.ndarray], seed) -> ClusteredDocuments:
    X = list(embeddings.values())
    best_k: int = silhouette_analysis(nr_labels, seed, X)
    model = KMeans(n_clusters=best_k, init='k-means++',
                   max_iter=300, n_init='auto')
    model.fit(X)

    # get the distance to the corresponding cluster centroid for each document
    centers = model.cluster_centers_
    clustered_documents: List[ClusteredDocuments] = []
    for idx in list(embeddings.keys()):
        vector = embeddings[idx]
        distances = [np.linalg.norm(center - vector) for center in centers]
        clustered_documents.append(ClusterDocument(
            idx, np.argmin(distances), np.min(distances)))
    clustered_obj = ClusteredDocuments(clustered_documents, len(centers))

    return clustered_obj


def get_docs_in_clusters(docs: List[int], clustered_docs: ClusteredDocuments) -> Dict[int, List[int]]:
    docs_by_cluster: Dict[int, List[int]] = {}
    for cluster in clustered_docs.clusters:
        docs_by_cluster[cluster] = []
    batch_docs: List[ClusterDocument] = clustered_docs.get_clustered_docs_by_idx(
        docs)
    for doc in batch_docs:
        docs_by_cluster[doc.cluster_idx].append(doc.idx)
    return docs_by_cluster


@TeacherRegistry.register("sequential-representation-least-confidence")
class SequentialRepresentationLCTeacher(BaseTeacher):
    """
    The sequential representative and least confidence teacher proposes samples that are most similar to unlabeled samples with least confidence in the predictions

    Applied to ER task:
        - Kholghi, M., De Vine, L., Sitbon, L., Zuccon, G., Nguyen, A.: Clinical information extraction using small data: An active learning approach based on sequence
        representations and word embeddings. Journal of the Association for Information Science and Technology 68(11), 2543–2556 (2017). doi: 10.1002/asi.23936
        (in paper AL strategie is called "Unsupervised Least Confidence" (ULC))
    """

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask, aggregation_method: Optional[AggregationMethod]):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task,
            aggregation_method=aggregation_method
        )
        self.nr_labels = len(self.labels)
        self.ngrams = NGramVectors([2, 3], seed)  # bi- and tri-ngrams
        self.embeddings: Dict[int, np.ndarray] = embed_documents_with_lexical_and_semantical_vectors(
            corpus=corpus, ngrams=self.ngrams)
        self.clustered_documents: ClusteredDocuments = cluster_documents(
            self.nr_labels, self.embeddings, seed)
        self.corpus = corpus

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        # get lc for documents (inside budget)
        batch = random.sample(potential_ids, budget)
        idx2text = self.corpus.get_text_by_ids(batch)
        prediction_results: Dict[int, PredictionResult] = self.predictor.predict(
            idx2text)
        lcs: Dict[int, float] = self.compute_function(
            prediction_results, step_size)

        # Get sorted dict for all clusters and their best docs in descending order
        # {cluster: [idx]} in descending order
        clustered_docs: Dict[int, List[int]] = get_docs_in_clusters(
            batch, self.clustered_documents)
        for key in list(clustered_docs.keys()):
            docs = clustered_docs[key]
            lc_docs = {doc: lcs[doc] for doc in docs}
            sorted_dict_by_lc = sorted(lc_docs.items(), key=lambda x: x[1])
            clustered_docs[key] = [key for key, value in sorted_dict_by_lc]

        # round robin for clusters, always take doc with lowest lc
        robin: int = 0  # Iterate over clusters
        num_clusters: int = len(self.clustered_documents.clusters)
        out_ids: List[int] = []

        while len(out_ids) < step_size:
            cur_cluster = robin % num_clusters
            # always add doc of cluster with max LC (lowest highest confidence)
            out_ids.append(clustered_docs[cur_cluster].pop(0))

        return out_ids

    def compute_ner(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, float]:
        """
        LC is calculated on token-level and aggregated on instance-level as configured.
        """
        scores = dict()
        for idx, prediction in predictions.items():
            token_confidences: List[TokenConfidence] = prediction.ner_confidences_token
            highest_confidences: List[float] = []
            for token in token_confidences:
                confidence_score: float = token.get_highest_confidence().confidence
                highest_confidences.append(confidence_score)
            instance_score = self.aggregate_function(highest_confidences)
            scores[idx] = instance_score
        return scores

    def compute_cls(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, int]:
        """ Not implemented
        """
        raise NotImplementedError(
            "Hybrid teacher is not implemented for text classification task.")
