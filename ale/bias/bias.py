from pathlib import Path
from typing import Dict, Any

import numpy as np
import srsly
from mlflow.entities import Run

from ale.bias.data_distribution import DataDistribution
from ale.bias.utils import normalize_counts
from ale.config import NLPTask
from ale.metrics.accuracy import Accuracy
from ale.trainer.prediction_result import PredictionResult


class BiasDetector:
    def __init__(self, nlp_task: NLPTask, label_column: str, file_raw: Path, ids=None):
        self.data_distribution = DataDistribution(nlp_task, label_column, file_raw)

        def filter_func(entry):
            if ids:
                return entry["id"] in ids
            else:
                return True

        self.corpus_by_id: Dict[int, Any] = {e["id"]: e for e in srsly.read_jsonl(file_raw) if filter_func(e)}
        self.ids = ids
        self.accuracy = Accuracy(nlp_task)

    def compute_and_log_distribution(self, mlflow_run: Run, artifact_name: str):
        if self.ids:
            distribution = self.data_distribution.get_data_distribution_by_label_for_ids(self.ids)
        else:
            distribution = self.data_distribution.get_data_distribution_by_label()

        self.data_distribution.store_distribution(distribution, mlflow_run, artifact_name)

        return distribution

    def compute_bias(self, distribution: Dict[str, float],
                     predictions: Dict[int, PredictionResult],
                     label_column: str):
        accuracy_per_label, error_per_label = self.accuracy(self.corpus_by_id, label_column, predictions)

        norm_distribution = normalize_counts(distribution)
        eps = 0.000000001

        bias = {label: error * -np.log(norm_distribution[label] + eps) for label, error in error_per_label.items() if label != 'O'}
        bias_by_optimum = {label: error * -np.log(abs(1/len(norm_distribution) - norm_distribution[label]) + eps) for label, error in error_per_label.items() if label != 'O'}
        bias_by_distribution_diff = {label: error * (1 + abs(1/len(norm_distribution) - norm_distribution[label])) for label, error in error_per_label.items() if label != 'O'}
        return accuracy_per_label, bias, error_per_label, bias_by_optimum, bias_by_distribution_diff

