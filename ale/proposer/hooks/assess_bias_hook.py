import logging
from pathlib import Path
from typing import List, Optional, Dict

import srsly
from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

import ale.mlflowutils.mlflow_utils as utils
from ale.bias.bias import BiasDetector
from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.proposer.hooks.abstract_hook import ProposeHook
from ale.trainer.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


class AssessBiasHook(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, **kwargs)
        self.iteration_counter_for_bias_assessment = 1

    @override
    def on_iter_end(self) -> None:
        self.iteration_counter_for_bias_assessment += 1

    @override
    def needs_dev_predictions(self) -> bool:
        return True

    @override
    def needs_train_predictions(self) -> bool:
        return True

    @override
    def after_prediction(self,
                         mlflow_run: Run,
                         preds_train: Optional[Dict[int, PredictionResult]],
                         preds_dev: Optional[Dict[int, PredictionResult]]) -> None:
        if self.iteration_counter_for_bias_assessment % self.cfg.experiment.assess_data_bias_eval_freq == 0:
            logger.info("Evaluate train data bias")
            self.assess_data_bias(mlflow_run, self.kwargs["train_file_raw"], "train", preds_train,
                                  ids=self.corpus.get_relevant_ids())

            logger.info("Evaluate dev data bias")
            self.assess_data_bias(mlflow_run, self.kwargs["dev_file_raw"], "dev", preds_dev)
            MlflowClient().set_tag(mlflow_run.info.run_id, "assess_data_bias", "True")
        else:
            logger.info(
                f"Skip data bias evaluation in iteration, interval: ({self.iteration_counter_for_bias_assessment}, {self.cfg.experiment.assess_data_bias_eval_freq})")

    def assess_data_bias(self, new_run, file_raw: Path,
                         prefix: str, preds: Optional[Dict[int, PredictionResult]], ids=None) -> None:
        """
        Compute data bias for the training dataset. The training dataset deserves a specific handling due to
        an increasing corpus over time.
        """
        bias_detector = BiasDetector(self.cfg.data.nlp_task, self.cfg.data.label_column, file_raw, ids=ids)
        distribution = bias_detector.compute_and_log_distribution(
            new_run,
            prefix + "/data_distribution")

        accuracy_per_label, bias, error_per_label, bias_by_optimum, bias_by_distribution_diff = bias_detector.compute_bias(
            distribution, preds,
            self.cfg.data.label_column)
        utils.store_bar_plot(accuracy_per_label, new_run, prefix + "/accuracy_per_label", ["Label", "Accuracy"])
        utils.store_bar_plot(error_per_label, new_run, prefix + "/error_per_label", ["Label", "Error"])
        utils.store_bar_plot(bias, new_run, prefix + "/bias", ["Label", "Bias"])
        utils.store_bar_plot(bias_by_optimum, new_run, prefix + "/bias_log_distr_diff", ["Label", "Bias"])
        utils.store_bar_plot(bias_by_distribution_diff, new_run, prefix + "/bias_distr_diff", ["Label", "Bias"])

        self.log_bias_metrics({
            "distribution": distribution,
            "bias": bias,
            "bias_log_distr_diff": bias_by_optimum,
            "bias_distr_diff": bias_by_distribution_diff,
            "accuracy": accuracy_per_label,
            "error": error_per_label
        }, prefix)

    def log_bias_metrics(self, metrics_to_log, data_split_prefix: str):
        for prefix, metrics in metrics_to_log.items():
            for label, value in metrics.items():
                MlflowClient().log_metric(
                    self.parent_run_id,
                    key=f"{data_split_prefix}_{prefix}_{label}",
                    value=value,
                    step=len(self.corpus)
                )
