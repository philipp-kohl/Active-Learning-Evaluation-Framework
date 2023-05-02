from abc import ABC
from typing import Dict

import numpy as np

from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.bandits.multi_armed_bandit import BanditTeacher


@TeacherRegistry.register("weighted-armed-bandit")
class WeightedBanditTeacher(BanditTeacher, ABC):
    def after_train(self, metrics: Dict):
        self.run_cnt += 1
        self.scores[self.curr_choice].append(
            2 * self.run_cnt * (metrics["cats_macro_f"] - self.last_choice)
        )
        self.last_choice = metrics["cats_macro_f"]
        mu = np.mean(self.scores[self.curr_choice])
        self.rewards[self.curr_choice] = mu
