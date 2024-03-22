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
        self.artifact_base_path = "confidence_assessment"

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
            logger.info("Evaluate train confidences")
            self.assess_confidence(mlflow_run, "train", preds_train)

            logger.info("Evaluate dev confidences")
            self.assess_confidence(mlflow_run, "dev", preds_dev)

            MlflowClient().set_tag(mlflow_run.info.run_id, "assess_confidence", "True")
        else:
            logger.info(
                f"Skip confidence evaluation in iteration, interval: ({self.iteration_counter_for_assessment}, {self.cfg.experiment.assess_overconfidence_eval_freq})")

    def build_artifact_path(self, data_split: str, folder_name: str):
        return f"{data_split}/{self.artifact_base_path}/{folder_name}"

    def assess_confidence(self, new_run,
                          prefix: str,
                          preds: Optional[Dict[int, PredictionResult]]) -> None:
        confidences = []
        true_positives = []
        for idx, doc_prediction in preds.items():
            for token_prediction in doc_prediction.ner_confidences_token:
                if token_prediction.gold_label == token_prediction.predicted_label:
                    true_positives.append(1)
                else:
                    true_positives.append(0)
                confidences.append(token_prediction.get_highest_confidence().confidence)

        ece_score = self.calculate_ece(confidences, true_positives, num_bins=10)
        self.plot_reliability_diagram_plotly(confidences, true_positives, new_run,
                                             self.build_artifact_path(prefix, "reliability_diagram"))
        utils.store_histogram(confidences, new_run, self.build_artifact_path(prefix, "confidences"),
                              ["Confidence", "Frequency"])

        MlflowClient().log_metric(
            self.parent_run_id,
            key=f"{prefix}_ece_overall",
            value=ece_score,
            step=len(self.corpus)
        )

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

    def plot_reliability_diagram_plotly(self, predicted_probabilities, true_labels, new_run, artifact_path, n_bins=10):
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

        bin_tp_frequency = np.zeros(n_bins)
        bin_confidence = np.zeros(n_bins)
        bin_count = np.zeros(n_bins)

        for b in range(n_bins):
            in_bin = bin_indices == b
            if np.sum(in_bin) > 0:
                bin_tp_frequency[b] = np.sum(true_labels[in_bin]) / len(true_labels[in_bin])
                bin_confidence[b] = np.mean(predicted_probabilities[in_bin])
                bin_count[b] = np.sum(in_bin)

        effective_bins = bin_count > 0

        # Plotting
        fig = go.Figure()
        # Add scatter plot for reliability
        fig.add_trace(go.Scatter(x=bin_confidence[effective_bins], y=bin_tp_frequency[effective_bins],
                                 mode='markers+lines', name='Model Calibration',
                                 error_y=dict(type='data', array=np.sqrt(
                                     bin_tp_frequency[effective_bins] * (1 - bin_tp_frequency[effective_bins]) /
                                     bin_count[
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

            utils.log_artifact(new_run, path, artifact_path=artifact_path)
