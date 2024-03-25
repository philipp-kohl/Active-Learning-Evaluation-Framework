from mlflow.artifacts import download_artifacts

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Any, Dict

import srsly
from mlflow.entities import Run
import ale.mlflowutils.mlflow_utils as mlflow_utils

from ale.config import TrainerConfig, AppConfig


class Corpus(ABC):

    ARTIFACT_FILE = "relevant_ids.json"

    def __init__(self, cfg: AppConfig, data_dir: Union[str, Path]):
        self.cfg = cfg
        self.data_dir = data_dir
        self.relevant_ids: List[int] = []

    def get_relevant_ids(self) -> List[int]:
        return self.relevant_ids

    def add_increment(self, ids: List[int]):
        same_ids = set(ids).intersection(set(self.relevant_ids))
        if len(same_ids) > 0:
            raise ValueError(f"Trying to add ids to corpus, which are already part of it: {same_ids}")

        self.relevant_ids.extend(ids)

    @abstractmethod
    def get_trainable_corpus(self) -> Any:
        pass

    @abstractmethod
    def get_not_annotated_data_points_ids(self) -> List[int]:
        pass

    def __len__(self):
        """
        :return: the length of the current corpus (measured by relevant_ids, not the whole corpus)
        """
        return len(self.relevant_ids)

    def store_to_artifacts(self, run: Run):
        mlflow_utils.log_dict_as_artifact(run, {"relevant_ids": self.relevant_ids}, self.ARTIFACT_FILE)

    def restore_from_artifacts(self, run: Run):
        downloaded_path = download_artifacts(run_id=run.info.run_id, artifact_path=self.ARTIFACT_FILE)
        json = srsly.read_json(downloaded_path)
        self.relevant_ids = json["relevant_ids"]

    def do_i_have_to_annotate(self):
        return len(self.get_not_annotated_data_points_ids()) > 0

    @abstractmethod
    def get_all_texts_with_ids(self) -> Dict[int, str]:
        pass

    @abstractmethod
    def get_text_by_ids(self, idxs: List[int]) -> Dict[int, str]:
        pass

    def get_text_by_id(self, idx: int) -> str:
        return self.get_text_by_ids([idx])[0]
