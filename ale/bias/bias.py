from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple

import numpy as np
import srsly
from mlflow.entities import Run

from ale.bias.data_distribution import DataDistribution
from ale.config import NLPTask
from ale.trainer.prediction_result import PredictionResult, Span


class BiasDetector:
    def __init__(self, nlp_task: NLPTask, label_column: str, file_raw: Path, ids=None):
        self.data_distribution = DataDistribution(nlp_task, label_column, file_raw)
        def filter_func(entry):
            if ids:
                return entry["id"] in ids
            else:
                return True

        self.corpus_by_id = {e["id"]: e for e in srsly.read_jsonl(file_raw) if filter_func(e)}
        self.ids = ids

        self.compute_function: Callable[
            [Dict[int, Any], str, Dict[int, PredictionResult]], Tuple[Dict[str, float], Dict[str, float]]] = {
            NLPTask.CLS: self.compute_cls,
            NLPTask.NER: self.compute_ner
        }[nlp_task]

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
        accuracy_per_label, error_per_label = self.compute_function(self.corpus_by_id, label_column, predictions)

        norm_distribution = self.normalize_counts(distribution)
        eps = 0.000000001

        bias = {label: error * -np.log(norm_distribution[label] + eps) for label, error in error_per_label.items()}
        bias_by_optimum = {label: error * -np.log(abs(1/len(norm_distribution) - norm_distribution[label]) + eps) for label, error in error_per_label.items()}
        bias_by_distribution_diff = {label: error * 1 + abs(1/len(norm_distribution) - norm_distribution[label]) for label, error in error_per_label.items()}
        return accuracy_per_label, bias, error_per_label, bias_by_optimum, bias_by_distribution_diff

    @staticmethod
    def normalize_counts(counts_dict):
        """
        Normalizes the counts in the given defaultdict by the total count of all labels.

        Parameters:
        - counts_dict (defaultdict): A defaultdict with label counts.

        Returns:
        - dict: A dictionary with the same keys as counts_dict, but with values normalized
                so that they sum to 1.
        """
        total_count = sum(counts_dict.values())
        if total_count == 0:
            return {}

        normalized_dict = {label: count / total_count for label, count in counts_dict.items()}
        return normalized_dict

    def get_highest_score_entry(self, scores_dict):
        """
        Returns the dictionary entry with the highest score.

        Parameters:
        - scores_dict (dict): A dictionary where the keys are labels and the values are scores.

        Returns:
        - tuple: The key-value pair with the highest score.
        """
        # Check if the dictionary is not empty
        if scores_dict:
            # Find the key with the highest value
            highest_entry = max(scores_dict.items(), key=lambda x: x[1])
            return highest_entry
        else:
            return None, None

    def compute_cls(self, corpus_dict: Dict[int, Any], label_column: str, preds: Dict[int, PredictionResult]):
        # Initialize counters for accuracy calculation with defaultdict
        label_counts = defaultdict(int)
        correct_predictions = defaultdict(int)
        error_predictions = defaultdict(int)
        # Iterate through each example
        for idx in corpus_dict.keys():
            gold_labels = corpus_dict[idx][label_column]
            pred_label = self.get_highest_score_entry(preds[idx].classification_confidences)[0]

            # Update counts
            label_counts[gold_labels] += 1

            if gold_labels == pred_label:
                correct_predictions[gold_labels] += 1
            else:
                error_predictions[gold_labels] += 1

        # Calculate accuracy per label
        accuracy_per_label: Dict[str, float] = {label: correct_predictions[label] / label_counts[label]
                                                for label in label_counts}
        error_per_label: Dict[str, float] = {label: error_predictions[label] / label_counts[label]
                                             for label in label_counts}

        return accuracy_per_label, error_per_label

    @staticmethod
    def is_full_match(gold: Tuple[int, int, str], pred: Span) -> bool:
        return gold[0] == pred.start and gold[1] == pred.end and gold[2] == pred.label

    @staticmethod
    def is_partial_match(gold: Tuple[int, int, str], pred: Span) -> bool:
        # Check for overlapping spans and matching labels
        return not (pred.end <= gold[0] or pred.start >= gold[1]) and gold[2] == pred.label

    def compute_ner(self, corpus_dict: Dict[int, Any], label_column: str, preds: Dict[int, PredictionResult]):
        # Initialize counters for accuracy calculation with defaultdict
        label_counts = defaultdict(int)
        correct_predictions = defaultdict(int)
        error_predictions = defaultdict(int)
        # Iterate through each example
        for idx in corpus_dict.keys():
            gold_entities = corpus_dict[idx][label_column]
            ner_preds = preds[idx].ner_confidences
            if ner_preds is None:
                if len(gold_entities) > 0:
                    raise Exception("Let's punish the missing preds!")
                else:
                    continue
            pred_entities = list(preds[idx].ner_confidences.keys())

            for gold in gold_entities:
                label_counts[gold[2]] += 1
                matched = self.is_match(gold, pred_entities)

                if matched:
                    correct_predictions[gold[2]] += 1
                else:
                    error_predictions[gold[2]] += 1

        # Calculate accuracy per label
        accuracy_per_label: Dict[str, float] = {label: correct_predictions[label] / label_counts[label]
                                                for label in label_counts}
        error_per_label: Dict[str, float] = {label: error_predictions[label] / label_counts[label]
                                             for label in label_counts}

        return accuracy_per_label, error_per_label

    def is_match(self, gold, pred_entities):
        for pred in pred_entities:
            if self.is_full_match(gold, pred):  # TODO partial match?
                return True

        return False
