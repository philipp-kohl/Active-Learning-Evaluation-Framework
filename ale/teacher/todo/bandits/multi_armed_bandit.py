from abc import ABC
from pathlib import Path
from typing import List, Dict

import numpy as np
import numpy.random
from sklearn.cluster import KMeans

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bases.cluster_abstract import ClusterBaseTeacher


@TeacherRegistry.register("multi-armed-bandit")
class BanditTeacher(ClusterBaseTeacher, ABC):
    def __init__(
        self, step_size: int, needs_model: bool, budget: int, data: Path, k=5, eps=0.05
    ):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        self.k = k
        self.scores = {i: [1] for i in range(self.k)}
        self.rewards = {i: 1 for i in range(self.k)}
        self.last_choice = 0
        self.curr_choice = 0
        self.model = KMeans(n_clusters=self.k, init="k-means++", max_iter=100, n_init=1)
        X = self.tfidf_vectorize(potential_ids=self.df.id)
        self.model.fit(X)
        self.cluster = {i: [] for i in range(self.k)}
        for j in range(len(self.model.labels_)):
            self.cluster[self.model.labels_[j]].append(j)

    def get_propose_ids(self, choice: int, potential_ids: List[int], size):
        cluster = self.df.iloc[self.cluster[choice]]
        potential_cluster_ids = cluster[cluster.id.isin(potential_ids)].id.values
        rem_size = min(size, len(potential_cluster_ids))
        propose_ids = np.random.choice(
            potential_cluster_ids, size=rem_size, replace=False
        )
        self.curr_choice = choice
        return propose_ids

    def propose(self, potential_ids: List[int]) -> List[int]:
        propose_ids = list()
        step_size = min(len(potential_ids), self.step_size)
        unseen = self.rewards.copy()

        while len(propose_ids) < step_size:
            choice = np.argmax(unseen)
            if choice.dtype == list:
                choice = choice[0]
            size = step_size - len(propose_ids)
            propose_ids += self.get_propose_ids(
                choice=choice, potential_ids=potential_ids, size=size
            )
            unseen.pop(choice)

        return propose_ids

    def after_train(self, metrics: Dict):
        self.scores[self.curr_choice].append(metrics["cats_macro_f"] - self.last_choice)
        self.last_choice = metrics["cats_macro_f"]
        mu = np.mean(self.scores[self.curr_choice])
        self.rewards[self.curr_choice] = mu
