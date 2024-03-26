from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List, Union

import srsly
from mlflow.entities import Run

import ale.mlflowutils.mlflow_utils as utils
from ale.config import NLPTask


class DataDistribution:
    def __init__(self, nlp_task: NLPTask, label_column: str, train_file_raw: Path):
        self.nlp_task = nlp_task
        self.label_column = label_column
        self.count_func = {
            NLPTask.CLS: self.count_func_cls,
            NLPTask.NER: self.count_func_ner
        }[self.nlp_task]
        self.train_file_raw = train_file_raw

    def get_data_distribution_by_label_for_ids(self, train_ids: List[int]) -> Dict[str, int]:
        labels: Dict[str, int] = defaultdict(lambda: 0)
        for entry in srsly.read_jsonl(self.train_file_raw):
            idx = entry["id"]
            if idx in train_ids:
                self.count_func(labels, entry[self.label_column])

        return labels

    def get_data_distribution_by_label(self) -> Dict[str, int]:
        labels: Dict[str, int] = defaultdict(lambda: 0)
        for entry in srsly.read_jsonl(self.train_file_raw):
            self.count_func(labels, entry[self.label_column])

        return labels

    def count_func_cls(self, labels: Dict[str, int], entry: Union[List[str], str]):
        if isinstance(entry, List):
            for label in entry:
                labels[label] += 1
        else:
            labels[entry] += 1

    def count_func_ner(self, labels: Dict[str, int], entry: List[Tuple[int, int, str]]):
        """
        "labels":[[4,23,"ORG"],[59,65,"MISC"],[94,101,"MISC"]]
        """
        for e in entry:
            label = e[2]
            labels[label] += 1

    def store_distribution(self, distribution: Dict[str, int], mlflow_run: Run, artifact_name: str):
        utils.store_bar_plot(distribution, mlflow_run, artifact_name, ['Label', 'Occurences'])
