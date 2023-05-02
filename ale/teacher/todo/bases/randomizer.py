import random
from abc import ABC
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bases.base_teacher import BaseTeacher


@TeacherRegistry.register("randomizer")
class RandomTeacher(BaseTeacher, ABC):
    """
    Random teacher: folds the data initially and makes random choices from iterated folds as propose is called
    """

    def __init__(
        self, step_size: int, needs_model: bool, budget: int, data: Path, seed, k=10
    ):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data
        )
        self.k = k
        self.fold = 0
        self.seed = seed
        indices = pd.read_json(self.data_path, lines=True).id
        shuffled = indices.sample(frac=1)
        random.seed(self.seed)
        self.folds = np.array_split(shuffled, self.k)

    def get_propose_ids(self, choice: int, potential_ids: List[int], size: int):
        fold = self.folds[choice]
        potential_fold_ids = [data_id for data_id in fold if data_id in potential_ids]
        rem_size = min(size, len(potential_fold_ids))
        propose_ids = random.sample(potential_fold_ids, k=rem_size)
        return propose_ids

    def propose(self, potential_ids: List[int]) -> (str, List[int]):
        propose_ids = list()
        unseen = list(range(self.k))
        step_size = min(len(potential_ids), self.step_size)

        while len(propose_ids) < step_size:
            choice = random.choice(unseen)
            size = step_size - len(propose_ids)
            propose_ids += self.get_propose_ids(
                choice=choice, potential_ids=potential_ids, size=size
            )
            unseen.remove(choice)

        return propose_ids

    def after_train(self, metrics: dict):
        self.fold += 1
        self.fold = self.fold % self.k
