import logging
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import umap
from numpy.linalg import norm
from scipy.sparse import spmatrix
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples

from ale.corpus.corpus import Corpus

logger = logging.getLogger(__name__)


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


def bert_vectorize(corpus: Corpus):
    """ Vectorizes the given data with BERT

    Args:
        - corpus (Corpus): The corpus.

    Returns:
        - sentence_embeddings (ndarray): ND-Array containing the BERT embeddings for the corpus.
    """
    data: Dict[int, str] = corpus.get_all_texts_with_ids()
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(list(data.values()))
    return sentence_embeddings


def get_cosine_similarity(vec_1: np.matrix, vec_2: np.matrix) -> float:
    a: np.ndarray = np.asarray(vec_1.T).reshape((vec_1.size,))
    b: np.ndarray = np.asarray(vec_2.T).reshape((vec_2.size,))
    norm_a: float = norm(a)
    norm_b: float = norm(b)
    if norm_a != 0 and norm_b != 0:
        return np.dot(a, b) / (norm(a) * norm(b))
    return np.dot(a, b)


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


def silhouette_analysis(nr_labels: int, seed: int, metric: str, embeddings) -> int:
    """ Analyses clustering with k different clusters between 2 and max(10,2*nr_labels). Plots silhouette graphs.

    Args:
        - nr_labels (int): Number of label classes
        - seed (int): Seed for random initialization
        - metric (str): The metric to measure distance/similarity in the data space
        - embeddings (any): Embeddings for KMeans

    Returns:
        - k_best (int): K with highest silhouette score
    """
    ks: List[int] = np.arange(2, max(10, 2 * nr_labels))  # range from 2 to maximum of double the size of labels and 10
    best_k_with_score: Tuple[int, float] = [-1, -1]
    for k in ks:
        logger.info(f"Test KMeans for {k} clusters")
        model_test = KMeans(n_clusters=k, init='k-means++',
                            max_iter=300, n_init='auto', random_state=seed)
        model_prediction = model_test.fit_predict(embeddings)

        logger.info(f"Define silhouette score for {k} clusters")
        score = silhouette_score(embeddings, model_prediction)
        plot_silhouette(k, seed, metric, model_prediction, score, model_test, embeddings)
        if score > best_k_with_score[1]:
            best_k_with_score = [k, score]
    return best_k_with_score[0]


def plot_silhouette(k: int, seed: int, metric: str, model_prediction: np.ndarray, score: float, model: KMeans,
                    embeddings) -> None:
    """ Plots silhouette graph.

    Args:
        - k (int): Number of clusters
        - seed (int): Seed for random state
        - metric (str): The metric to measure distance/similarity in the data space
        - model_prediction (ndarray): The predictions of the clustering model
        - score (float): The silhouette score
        - model (KMeans): The KMeans model
        - embeddings (any): The embeddings for KMeans
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(embeddings) + (k + 1) * 10])
    sample_silhouette_values = silhouette_samples(embeddings, model_prediction)
    y_lower: int = 10
    for i in range(k):
        ith_cluster_values = sample_silhouette_values[model_prediction == i]
        ith_cluster_values.sort()
        ith_cluster_size: int = ith_cluster_values.shape[0]
        y_upper = y_lower + ith_cluster_size
        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the plots with their cluster numbers
        ax1.text(-0.05, y_lower + 0.5 * ith_cluster_size, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot for " + str(k) + " clusters")
    ax1.set_xlabel("Coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=score, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    logger.info(f"Compute UMAP for embeddings")
    X_embedded = umap.UMAP(n_components=2, n_neighbors=10, metric="cosine", densmap=True).fit_transform(embeddings)

    colors = cm.nipy_spectral(model_prediction.astype(float) / k)
    ax2.scatter(
        X_embedded[:, 0], X_embedded[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = model.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" %
                                       i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("Clustered data with " + str(k) + " clusters")
    ax2.set_xlabel("Feature space of 1st feature")
    ax2.set_ylabel("Feature space of 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with k = %d"
        % k,
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(f"./silhouette_plots_debug_{k}.png")


class ClusterDocument:
    def __init__(self, idx: int, cluster_idx: int, distance: float):
        self.idx = idx
        self.cluster_idx = cluster_idx
        self.distance = distance


class ClusteredDocuments:
    def __init__(self, documents: List[ClusterDocument], num_clusters: int):
        self.clusters = np.arange(0, num_clusters)
        self.documents = documents

    def get_clustered_docs_by_idx(self, indices: List[int]) -> List[ClusterDocument]:
        output = [doc for doc in self.documents if doc.idx in indices]
        return output

