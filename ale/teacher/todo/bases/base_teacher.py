from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict


class BaseTeacher(ABC):
    """
    abstract teacher class
    """

    def __init__(self, step_size: int, needs_model: bool, budget: int, data: Path):
        self.step_size = step_size
        self.needs_model = needs_model
        self.budget = budget
        self.data_path = data

    @abstractmethod
    def propose(self, potential_ids: List[int]) -> List[int]:
        """
        :type potential_ids: object
        :return: data_uri and List of data indices
        """
        pass

    def after_train(self, metrics: Dict):
        pass

    def after_initial_train(self, metrics: Dict):
        pass
