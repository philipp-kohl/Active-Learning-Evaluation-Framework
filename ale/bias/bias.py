from pathlib import Path
from typing import List

from mlflow.entities import Run

from ale.bias.data_distribution import DataDistribution
from ale.config import NLPTask


class BiasDetector:
    def __init__(self, nlp_task: NLPTask, label_column: str, train_file_raw: Path):
        self.data_distribution = DataDistribution(nlp_task, label_column, train_file_raw)

    def compute_and_log_distribution(self, train_ids: List[int], mlflow_run: Run, artifact_name: str):
        distribution = self.data_distribution.get_data_distribution_by_label_for_ids(train_ids)
        self.data_distribution.store_distribution(distribution, mlflow_run, artifact_name)


