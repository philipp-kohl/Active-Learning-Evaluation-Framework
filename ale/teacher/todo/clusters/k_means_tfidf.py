from abc import ABC

from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from pathlib import Path
from sklearn.cluster import KMeans

from ale.trainer.base_trainer import Predictor

"""
These are naive approaches to perform active learning via text clustering and making random choices of equal size 
from each cluster
"""


@TeacherRegistry.register("k-means-tfidf")
class KMeansTfidfTeacher(KMeansBase, ABC):

    def __init__(self, step_size: int, needs_model: bool, budget: int, data: Path, k=2):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        self.k = k
        self.model = KMeans(n_clusters=self.k, init="k-means++", max_iter=300, n_init=5)
        X = self.tfidf_vectorize(potential_ids=self.df.id)
        self.model.fit(X)
        self.cluster = {i: [] for i in range(self.k)}
        for j in range(len(self.model.labels_)):
            self.cluster[self.model.labels_[j]].append(j)
