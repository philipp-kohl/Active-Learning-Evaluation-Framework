import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import srsly
from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

import ale.mlflowutils.mlflow_utils as utils
from ale.bias.bias import BiasDetector
from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.metrics.accuracy import Accuracy
from ale.proposer.hooks.abstract_hook import ProposeHook
from ale.trainer.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


class AssessConfidenceHook(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, **kwargs)
        self.iteration_counter_for_assessment = 1
        self.accuracy = Accuracy(cfg.data.nlp_task)

    @override
    def on_iter_end(self) -> None:
        self.iteration_counter_for_assessment += 1

    @override
    def needs_dev_predictions(self) -> bool:
        return True

    @override
    def after_prediction(self,
                         mlflow_run: Run,
                         preds_train: Optional[Dict[int, PredictionResult]],
                         preds_dev: Optional[Dict[int, PredictionResult]]) -> None:
        if self.iteration_counter_for_assessment % self.cfg.experiment.assess_overconfidence_eval_freq == 0:
            # logger.info("Evaluate train confidences")
            # self.assess_confidence(mlflow_run, self.kwargs["train_file_raw"], "train", preds_train,
            #                        ids=self.corpus.get_relevant_ids())

            logger.info("Evaluate dev confidences")
            self.assess_confidence_v2(mlflow_run, self.kwargs["dev_file_raw"], "dev", preds_dev)
            MlflowClient().set_tag(mlflow_run.info.run_id, "assess_confidence", "True")
        else:
            logger.info(
                f"Skip confidence evaluation in iteration, interval: ({self.iteration_counter_for_assessment}, {self.cfg.experiment.assess_data_bias_eval_freq})")

    def assess_confidence(self, new_run, file_raw: Path,
                          prefix: str, preds: Optional[Dict[int, PredictionResult]], ids=None) -> None:

        corpus_by_id: Dict[int, Any] = {e["id"]: e for e in srsly.read_jsonl(file_raw)}
        acc, err = self.accuracy(corpus_by_id, self.cfg.data.label_column, preds)

        confidences: Dict[str, float] = defaultdict(float)
        for _, pred in preds.items():
            for span, score in pred.ner_confidences.items():
                confidences[span.label] = score

        ece_per_label = {}

        for label, acc in acc.items():
            avgConf = np.mean(confidences[label])
            ece_per_label[label] = abs(acc - avgConf)

        utils.store_bar_plot(ece_per_label, new_run, prefix + "/ece", ["Label", "ECE"])

        self.log_bias_metrics({
            "ece": ece_per_label,
        }, prefix)

    def assess_confidence_v2(self, new_run, file_raw: Path,
                             prefix: str, preds: Optional[Dict[int, PredictionResult]], ids=None) -> None:

        label_column = self.cfg.data.label_column
        corpus_dict: Dict[int, Any] = {e["id"]: e for e in srsly.read_jsonl(file_raw)}

        labels = []
        confidences = []
        for idx in corpus_dict.keys():
            gold_entities = corpus_dict[idx][label_column]
            ner_preds = preds[idx].ner_confidences
            if ner_preds is None:
                if len(gold_entities) > 0:
                    raise Exception("Let's punish the missing preds!")
                else:
                    continue
            pred_entities = preds[idx].ner_confidences

            for gold in gold_entities:
                matched = False
                score = 0
                for span, score in pred_entities.items():
                    if Accuracy.is_full_match(gold, span):  # TODO partial match?
                        matched = True
                        break

                if matched:
                    labels.append(1)
                    confidences.append(score)
                else:
                    labels.append(0)
                    confidences.append(0)

        ece_score = self.calculate_ece(confidences, labels, num_bins=10)
        self.plot_reliability_diagram_plotly(confidences, labels, new_run)
        all_confidences = [confidences for idx in corpus_dict.keys() for span, confidences in
                           preds[idx].ner_confidences.items()]
        utils.store_histogram(all_confidences, new_run, prefix + "/confidences", ["Confidence", "Frequency"])

        MlflowClient().log_metric(
            self.parent_run_id,
            key=f"dev_ece_overall",
            value=ece_score,
            step=len(self.corpus)
        )
        # self.log_bias_metrics({
        #     "ece": ece_score,
        # }, prefix)

    def calculate_ece(self, confidences, true_labels, num_bins=15):
        """
        Compute the Expected Calibration Error (ECE).

        Args:
        - confidences (np.array): Array of confidence scores from the model.
        - true_labels (np.array): Array of the true labels.
        - num_bins (int): Number of bins to use for calibration error calculation.

        Returns:
        - ece (float): The expected calibration error.
        """
        confidences = np.array(confidences)
        true_labels = np.array(true_labels)

        # Verify the inputs
        assert len(confidences) == len(true_labels), "The number of confidence scores and labels must be the same"
        assert all(0 <= np.array(confidences)) and all(
            np.array(confidences) <= 1), "Confidences must be between 0 and 1"
        assert num_bins > 0, "Number of bins must be positive"

        # Initialize the ECE
        ece = 0.0

        # Define bin edges and bin width
        bin_edges = np.linspace(0, 1, num_bins + 1)

        # Calculate ECE
        for i in range(num_bins):
            # Get the confidences and corresponding true labels in the bin
            bin_confidences = confidences[(confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])]
            bin_labels = true_labels[(confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])]

            # If the bin is not empty, calculate its contribution to the ECE
            if bin_confidences.size > 0:
                accuracy_in_bin = np.mean(bin_labels)
                avg_confidence_in_bin = np.mean(bin_confidences)
                bin_error = np.abs(accuracy_in_bin - avg_confidence_in_bin)

                # Weight by the number of samples in the bin
                bin_weight = len(bin_confidences) / len(confidences)

                # Update the ECE
                ece += bin_error * bin_weight

        return ece

    def log_bias_metrics(self, metrics_to_log, data_split_prefix: str):
        for prefix, metrics in metrics_to_log.items():
            for label, value in metrics.items():
                MlflowClient().log_metric(
                    self.parent_run_id,
                    key=f"{data_split_prefix}_{prefix}_{label}",
                    value=value,
                    step=len(self.corpus)
                )

    import numpy as np
    import plotly.graph_objects as go

    def plot_reliability_diagram_plotly(self, predicted_probabilities, true_labels, new_run, n_bins=10):
        """
        Plots a reliability diagram using Plotly for binary classification predictions.

        Parameters:
        - predicted_probabilities: array-like, predicted probabilities of the positive class.
        - true_labels: array-like, true binary labels (0 or 1) of the same length as predicted_probabilities.
        - n_bins: int, the number of bins to use.
        """
        import plotly.graph_objects as go
        predicted_probabilities = np.array(predicted_probabilities)
        true_labels = np.array(true_labels)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probabilities, bins) - 1

        bin_accuracy = np.zeros(n_bins)
        bin_confidence = np.zeros(n_bins)
        bin_count = np.zeros(n_bins)

        for b in range(n_bins):
            in_bin = bin_indices == b
            if np.sum(in_bin) > 0:
                bin_accuracy[b] = np.sum(true_labels[in_bin]) / len(true_labels[in_bin])
                bin_confidence[b] = np.mean(predicted_probabilities[in_bin])
                bin_count[b] = np.sum(in_bin)

        effective_bins = bin_count > 0

        # Plotting
        fig = go.Figure()
        # Add scatter plot for reliability
        fig.add_trace(go.Scatter(x=bin_confidence[effective_bins], y=bin_accuracy[effective_bins],
                                 mode='markers+lines', name='Model Calibration',
                                 error_y=dict(type='data', array=np.sqrt(
                                     bin_accuracy[effective_bins] * (1 - bin_accuracy[effective_bins]) / bin_count[
                                         effective_bins]),
                                              visible=True)))
        # Add line for perfect calibration
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', line=dict(dash='dash')))

        fig.update_layout(title='Reliability Diagram',
                          xaxis_title='Predicted Probability',
                          yaxis_title='Relative Frequency (TP)',
                          yaxis=dict(scaleanchor="x", scaleratio=1),
                          xaxis=dict(constrain='domain'),
                          showlegend=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = f'{temp_dir}/reliability_diagram.html'
            fig.write_html(path)

            utils.log_artifact(new_run, path, artifact_path="dev/reliability_diagram")

    # Example usage:
    # Replace `predicted_probabilities` and `true_labels` with your actual data arrays.
    # predicted_probabilities = np.array([...])
    # true_labels = np.array([...])
    # plot_reliability_diagram_plotly(predicted_probabilities, true_labels)

