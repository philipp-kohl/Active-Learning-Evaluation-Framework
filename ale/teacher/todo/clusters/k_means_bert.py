from abc import ABC

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bases.cluster_abstract import ClusterBaseTeacher
from pathlib import Path
from sklearn.cluster import KMeans


@TeacherRegistry.register("k-means-bert")
class KMeansBertTeacher(ClusterBaseTeacher, ABC):
    def __init__(self, step_size: int, needs_model: bool, budget: int, data: Path, k=2):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        self.k = k
        self.model = KMeans(n_clusters=self.k, init="k-means++", max_iter=300, n_init=5)
        X = self.bert_vectorize(potential_ids=self.df.id)
        self.model.fit(X)
        self.cluster = {i: [] for i in range(self.k)}
        for j in range(len(self.model.labels_)):
            self.cluster[self.model.labels_[j]].append(j)
