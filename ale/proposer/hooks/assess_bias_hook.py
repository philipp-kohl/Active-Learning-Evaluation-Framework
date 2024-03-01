import logging
from pathlib import Path

import srsly
from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

import ale.mlflowutils.mlflow_utils as utils
from ale.bias.bias import BiasDetector
from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.proposer.hooks.abstract_hook import ProposeHook

logger = logging.getLogger(__name__)


class AssessBiasHook(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, **kwargs)
        self.iteration_counter_for_bias_assessment = 1

    @override
    def after_training(self, mlflow_run: Run) -> None:
        if self.cfg.experiment.assess_data_bias:
            if self.iteration_counter_for_bias_assessment % self.cfg.experiment.assess_data_bias_eval_freq == 0:
                logger.info("Evaluate train data bias")
                self.assess_data_bias_train(mlflow_run, self.kwargs["train_file_raw"], "train/")

                logger.info("Evaluate dev data bias")
                self.assess_data_bias(mlflow_run, self.kwargs["dev_file_raw"], "dev/")
                MlflowClient().set_tag(mlflow_run.info.run_id, "assess_data_bias", "True")
            else:
                logger.info(
                    f"Skip data bias evaluation in iteration, interval: ({self.iteration_counter_for_bias_assessment}, {self.cfg.experiment.assess_data_bias_eval_freq})")

    @override
    def on_iter_end(self) -> None:
        self.iteration_counter_for_bias_assessment += 1

    def assess_data_bias_train(self, new_run, file_raw: Path,
                               artifact_dir: str) -> None:
        """
        Compute data bias for the training dataset. The training dataset deserves a specific handling due to
        an increasing corpus over time.
        """
        bias_detector = BiasDetector(self.cfg.data.nlp_task, self.cfg.data.label_column, file_raw)
        distribution = bias_detector.compute_and_log_distribution(
            new_run,
            artifact_dir + "data_distribution",
            ids=self.corpus.get_relevant_ids())

        corpus = []
        for entry in srsly.read_jsonl(file_raw):
            assert entry["id"] is not None
            if entry["id"] in self.corpus.get_relevant_ids():
                corpus.append(entry)

        bias, accuracy_per_label, error_per_label = self.predict_and_compute_bias(artifact_dir, bias_detector, corpus,
                                                                                  distribution, new_run)
        self.log_bias_metrics(accuracy_per_label, bias, distribution, error_per_label, "train")

    def assess_data_bias(self, new_run, file_raw: Path, artifact_dir: str) -> None:
        bias_detector = BiasDetector(self.cfg.data.nlp_task, self.cfg.data.label_column, file_raw)
        distribution = bias_detector.compute_and_log_distribution(
            new_run,
            artifact_dir + "data_distribution")

        corpus = []
        for idx, entry in enumerate(srsly.read_jsonl(file_raw)):
            entry["id"] = idx
            corpus.append(entry)

        bias, accuracy_per_label, error_per_label = self.predict_and_compute_bias(artifact_dir, bias_detector, corpus,
                                                                                  distribution, new_run)

        self.log_bias_metrics(accuracy_per_label, bias, distribution, error_per_label, "dev")

    def log_bias_metrics(self, accuracy_per_label, bias, distribution, error_per_label, data_split_prefix):
        metrics_to_log = {
            "distribution": distribution,
            "bias": bias,
            "accuracy": accuracy_per_label,
            "error": error_per_label
        }
        for prefix, metrics in metrics_to_log.items():
            for label, value in metrics.items():
                MlflowClient().log_metric(
                    self.parent_run_id,
                    key=f"{data_split_prefix}_{prefix}_{label}",
                    value=value,
                    step=len(self.corpus)
                )

    def predict_and_compute_bias(self, artifact_dir, bias_detector: BiasDetector, corpus, distribution, new_run):
        preds = self.kwargs["trainer"].predict({e["id"]: e["text"] for e in
                                      corpus})  # TODO extract to outer scope? Predict and train and dev and deliver to hooks

        accuracy_per_label, bias, error_per_label = bias_detector.compute_bias({e["id"]: e for e in corpus},
                                                                               distribution, preds,
                                                                               self.cfg.data.label_column)
        utils.store_bar_plot(accuracy_per_label, new_run, artifact_dir + "accuracy_per_label", ["Label", "Accuracy"])
        utils.store_bar_plot(error_per_label, new_run, artifact_dir + "error_per_label", ["Label", "Error"])
        utils.store_bar_plot(bias, new_run, artifact_dir + "bias", ["Label", "Bias"])

        return bias, accuracy_per_label, error_per_label