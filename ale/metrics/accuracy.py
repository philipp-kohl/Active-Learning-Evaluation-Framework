from collections import defaultdict
from typing import Callable, Dict, Tuple, Any

from ale.utils import NLPTask
from ale.trainer.prediction_result import PredictionResult, Span


class Accuracy:
    def __init__(self, nlp_task: NLPTask):
        self.compute_function: Callable[
            [Dict[int, Any], str, Dict[int, PredictionResult]], Tuple[Dict[str, float], Dict[str, float]]] = {
            NLPTask.CLS: self.compute_cls,
            NLPTask.NER: self.compute_ner
        }[nlp_task]

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
    def get_highest_score_entry(scores_dict):
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

        for _, doc_pred in preds.items():
            for token_pred in doc_pred.ner_confidences_token:
                gold_label = token_pred.gold_label.lstrip("B-").lstrip("I-")
                label_counts[gold_label] += 1

                if token_pred.gold_label == token_pred.predicted_label:
                    correct_predictions[gold_label] += 1
                else:
                    error_predictions[gold_label] += 1

        # Calculate accuracy per label
        accuracy_per_label: Dict[str, float] = {label: correct_predictions[label] / label_counts[label]
                                                for label in label_counts}
        error_per_label: Dict[str, float] = {label: error_predictions[label] / label_counts[label]
                                             for label in label_counts}

        return accuracy_per_label, error_per_label

    @staticmethod
    def is_match(gold, pred_entities):
        for pred in pred_entities:
            if Accuracy.is_full_match(gold, pred):  # TODO partial match?
                return True

        return False

    def __call__(self, examples: Dict[int, Any], label_column: str, predictions: Dict[int, PredictionResult]) -> Tuple[
        Dict[str, float], Dict[str, float]]:
        return self.compute_function(examples, label_column, predictions)
