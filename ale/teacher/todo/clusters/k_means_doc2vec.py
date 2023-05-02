from abc import ABC

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bases.cluster_abstract import KMeansBase
from pathlib import Path


@TeacherRegistry.register("k-means-doc2vec")
class KMeansDoc2VecTeacher(KMeansBase, ABC):
    def __init__(self, step_size: int, needs_model: bool, budget: int, data: Path, k=2):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        X = self.doc2vec_vectorize(potential_ids=self.df.id)
        self.model.fit(X)
        self.cluster = {i: [] for i in range(self.k)}
        for j in range(len(self.model.labels_)):
            self.cluster[self.model.labels_[j]].append(j)
