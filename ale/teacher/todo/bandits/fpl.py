from abc import ABC
from pathlib import Path
from typing import Dict

import numpy as np

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bandits.multi_armed_bandit import BanditTeacher


@TeacherRegistry.register("fpl")
class FollowPertubatedLeader(BanditTeacher, ABC):
    """
    FPL Algorithm for adversarial bandit.
    In this variant, at each iteration, the teacher
    chooses an arm and an adversary simultaneously
    chooses the payoff structure for each arm.
    This is one of the strongest generalizations of the bandit problem.
    """

    # TODO: rethink self.rewards
    def __init__(
        self, step_size: int, needs_model: bool, budget: int, data: Path, scale=1.0
    ):
        super().__init__(
            step_size=step_size, needs_model=needs_model, budget=budget, data=data, k=2
        )
        self.scale = scale
        self.scores = {i: [0] for i in range(self.k)}
        self.rewards = {i: 0 for i in range(self.k)}

    def after_train(self, metrics: Dict):
        noise = np.random.exponential(scale=self.scale, size=self.k)
        self.scores[self.curr_choice].append(metrics["cats_macro_f"] - self.last_choice)
        self.last_choice = metrics["cats_macro_f"]
        for i in range(self.k):
            self.rewards[i] = np.mean(self.scores[i]) + noise[i]
