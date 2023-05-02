from abc import ABC
from pathlib import Path
from typing import List, Dict

import numpy as np

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bandits.multi_armed_bandit import BanditTeacher


@TeacherRegistry.register("exp3")
class Exp3(BanditTeacher, ABC):
    """
    Exp3 chooses an arm at random with probability ( 1 − γ )
    it prefers arms with higher weights (exploit),
    it chooses with probability γ to uniformly randomly explore.
    After receiving the rewards the weights are updated.
    The exponential growth significantly increases the weight of good arms.
    """

    def __init__(
        self, step_size: int, needs_model: bool, budget: int, data: Path, scale=1.0
    ):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data, k=2
        )
        self.scale = scale
        self.p = [
            (1 - self.scale) + 1 / self.k + self.scale / self.k for _ in range(self.k)
        ]
        self.rewards = np.zeros(self.k)
        self.w = np.ones(self.k)

    def propose(self, potential_ids: List[int]) -> List[int]:
        propose_ids = list()
        unseen = list(range(self.k))

        while len(propose_ids) < self.step_size:
            choice = np.random.choice(unseen, p=self.p, size=1)[0]
            size = self.step_size - len(propose_ids)
            propose_ids += self.get_propose_ids(
                choice=choice, potential_ids=potential_ids, size=size
            )
            unseen.remove(choice)

        return propose_ids

    def after_train(self, metrics: Dict):
        reward = metrics["cats_macro_f"] - self.last_choice
        self.last_choice = metrics["cats_macro_f"]
        self.rewards = np.zeros(self.k)
        self.rewards[self.curr_choice] = reward / self.p[self.curr_choice]
        for j in range(self.k):
            self.w[j] *= np.exp(self.scale * self.rewards[j] / self.k)
        self.p = [
            (1 - self.scale) + self.w[i] / np.sum(self.w) + self.scale / self.k
            for i in range(self.k)
        ]
