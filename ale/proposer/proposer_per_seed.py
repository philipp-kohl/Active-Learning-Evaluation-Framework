import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict

import numpy as np
import srsly
from mlflow import MlflowClient
from mlflow.entities import RunStatus, Run
from mlflow.utils import mlflow_tags

import ale.mlflowutils.mlflow_utils as utils
from ale.bias.bias import BiasDetector
from ale.config import AppConfig
from ale.corpus.corpus import Corpus

from ale.registry.registerable_corpus import CorpusRegistry
from ale.registry.registerable_teacher import TeacherRegistry
from ale.registry.registerable_trainer import TrainerRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.base_trainer import MetricsType, PredictionTrainer

logger = logging.getLogger(__name__)


class AleBartenderPerSeed:

    def __init__(self,
                 cfg: AppConfig,
                 seed: int,
                 train_file_converted: Path,
                 dev_file_converted: Path,
                 test_file_converted: Path,
                 train_file_raw: Path,
                 dev_file_raw: Path,
                 labels: List[Any],
                 experiment_id: str,
                 parent_run_id: str,
                 tracking_metrics: List[str]):
        self.cfg = cfg
        self.seed = seed
        self.experiment_id = experiment_id
        self.parent_run_id = parent_run_id
        self.tracking_metrics = tracking_metrics

        logger.info(f"Use corpus manager: {self.cfg.trainer.corpus_manager}")
        corpus_class = CorpusRegistry.get_instance(self.cfg.trainer.corpus_manager)
        self.corpus = corpus_class(train_file_converted)
        logger.info(f"Use trainer: {self.cfg.trainer.trainer_name}")
        trainer_class = TrainerRegistry.get_instance(
            self.cfg.trainer.trainer_name
        )
        self.trainer: PredictionTrainer = trainer_class(
            dev_file_converted,
            test_file_converted,
            Path(self.cfg.trainer.config_path),
            self.cfg.technical.use_gpu,
            seed,
            self.cfg.data.nlp_task,
            self.cfg.trainer.recreate_pipeline_each_run
        )
        logger.info(f"Use strategy: {self.cfg.teacher.strategy}")
        teacher_strategy_class = TeacherRegistry.get_instance(
            self.cfg.teacher.strategy
        )
        self.teacher: BaseTeacher = teacher_strategy_class(
            corpus=self.corpus,
            predictor=self.trainer,
            seed=seed,
            labels=labels
        )
        logger.info(f"Use '{self.cfg.experiment.initial_data_strategy}' teacher for initial data ratio.")
        initial_teacher_strategy_class = TeacherRegistry.get_instance(
            self.cfg.experiment.initial_data_strategy
        )
        self.initial_teacher: BaseTeacher = initial_teacher_strategy_class(
            corpus=self.corpus,
            predictor=self.trainer,
            seed=seed,
            labels=labels
        )
        self.bias_detector_train = BiasDetector(self.cfg.data.nlp_task, self.cfg.data.label_column, train_file_raw)
        self.bias_detector_dev = BiasDetector(self.cfg.data.nlp_task, self.cfg.data.label_column, dev_file_raw)
        self.dev_file_raw = dev_file_raw
        self.train_file_raw = train_file_raw
        self.train_file_converted = train_file_converted

    def run_single_seed(self) -> None:
        """
        Run the expiremnt for a single seed. Setup the internal random state with the given seed.
        Starts the initial training on the first increment of not annotated data points,
        then continues with the teacher-proposing and training process.
        :param seed: Seed to run the experiment with.
        :return None
        """
        logger.info(f"Start seed: {self.seed}")

        initial_data_ratio = self.cfg.experiment.initial_data_ratio
        all_ids = self.corpus.get_not_annotated_data_points_ids()
        first_step_size = int(initial_data_ratio * len(all_ids))
        logger.info(f"Initial step size: {first_step_size}")

        old_run: Optional[Run] = None
        already_runned_runs = utils.get_all_child_runs(self.experiment_id, self.parent_run_id, RunStatus.FINISHED)
        if already_runned_runs:
            newest_run = already_runned_runs[0]
            logger.info(f"Restore run: {newest_run.info.run_name}")
            self.trainer.restore_from_artifacts(newest_run)
            self.corpus.restore_from_artifacts(newest_run)
            old_run = newest_run
        else:
            initial_data_ids = self.initial_teacher.propose(all_ids, first_step_size, self.cfg.teacher.sampling_budget)
            self.corpus.add_increment(initial_data_ids)
            initial_train_evaluation_metrics, _, run = self.initial_train(self.corpus, self.seed)
            old_run = run
            self.teacher.after_initial_train(initial_train_evaluation_metrics)

        annotation_budget: int = self.cfg.experiment.annotation_budget
        iteration_counter_for_bias_assessment = 1
        while self.corpus.do_i_have_to_annotate():
            if len(self.corpus) >= annotation_budget:
                logger.info(f"Stop seed run due to exceeded annotation budget ({annotation_budget})")
                break

            self.propose_new_data(self.corpus)

            evaluation_metrics, test_metrics, new_run = self.train(
                self.corpus, f"train {len(self.corpus)}", self.seed
            )

            if self.cfg.experiment.assess_data_bias:
                if iteration_counter_for_bias_assessment % self.cfg.experiment.assess_data_bias_eval_freq == 0:
                    logger.info("Evaluate train data bias")
                    self.assess_data_bias_train(self.bias_detector_train, new_run, self.train_file_raw,
                                                "train/")

                    logger.info("Evaluate dev data bias")
                    self.assess_data_bias(self.bias_detector_dev, new_run, self.dev_file_raw,
                                          "dev/")
                    MlflowClient().set_tag(new_run.info.run_id, "assess_data_bias", "True")
                else:
                    logger.info(
                        f"Skip data bias evaluation in iteration, interval: ({iteration_counter_for_bias_assessment}, {self.cfg.experiment.assess_data_bias_eval_freq})")

            # Delete artifacts for old run. We do not need them for resume
            self.trainer.delete_artifacts(old_run)
            old_run = new_run
            self.teacher.after_train(evaluation_metrics)
            iteration_counter_for_bias_assessment += 1
        logger.info("End seed: %s", self.seed)

    def assess_data_bias_train(self, bias_detector: BiasDetector, new_run, file_raw: Path,
                               artifact_dir: str) -> None:
        """
        Compute data bias for the training dataset. The training dataset deserves a specific handling due to
        an increasing corpus over time.
        """
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

    def assess_data_bias(self, bias_detector: BiasDetector, new_run, file_raw: Path,
                         artifact_dir: str) -> None:
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
        preds = self.trainer.predict({e["id"]: e["text"] for e in corpus})

        accuracy_per_label, bias, error_per_label = bias_detector.compute_bias({e["id"]: e for e in corpus},
                                                                               distribution, preds,
                                                                               self.cfg.data.label_column)
        utils.store_bar_plot(accuracy_per_label, new_run, artifact_dir + "accuracy_per_label", ["Label", "Accuracy"])
        utils.store_bar_plot(error_per_label, new_run, artifact_dir + "error_per_label", ["Label", "Error"])
        utils.store_bar_plot(bias, new_run, artifact_dir + "bias", ["Label", "Bias"])

        return bias, accuracy_per_label, error_per_label

    def train(self, corpus: Corpus, run_name: str, seed: int) -> Tuple[MetricsType, MetricsType, Run]:
        """
        Starts a new run with the given corpus and trains the model on it.
        Returns the evaluation metrics of the run.
        :param corpus: Corpus to train on
        :param run_name: Name of the run
        :param seed: Seed of the run
        :return: Evaluation metrics of the run.
        :raises ValueError: If the corpus is empty.
        :raises ValueError: If the corpus is not annotated.
        """
        seed_tag = {"seed": str(seed)}
        tags = seed_tag.copy()
        tags[mlflow_tags.MLFLOW_PARENT_RUN_ID] = self.parent_run_id
        tags[mlflow_tags.MLFLOW_GIT_COMMIT] = self.cfg.mlflow.git_hash
        tags[mlflow_tags.MLFLOW_USER] = self.cfg.mlflow.user
        tags[mlflow_tags.MLFLOW_SOURCE_NAME] = self.cfg.mlflow.source_name
        tags[mlflow_tags.MLFLOW_SOURCE_TYPE] = "LOCAL"
        logger.info(f"Train with new increment ({len(corpus)})")
        logger.info(f"Start child run: {run_name}")
        train_run = MlflowClient().create_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
        )
        try:
            utils.walk_params_from_omegaconf_dict(
                self.cfg, lambda name, value: utils.log_param(train_run, name, value)
            )
            evaluation_metrics = self.trainer.train(corpus, train_run)
            test_metrics = self.test_and_log(corpus)
            self.trainer.store_to_artifacts(train_run)
            corpus.store_to_artifacts(train_run)
            utils.mark_run_as_finished(train_run, RunStatus.FINISHED)

            return evaluation_metrics, test_metrics, train_run
        except Exception as e:
            utils.mark_run_as_finished(train_run, RunStatus.FAILED)
            raise e

    def initial_train(self, corpus: Corpus, seed: int) -> Tuple[MetricsType, MetricsType, Run]:
        """
        Initial training call. Starts a new run with the given corpus and trains the model on it.
        If a run with the same seed and corpus already exists, the run is restored.
        Returns the evaluation metrics of the run.
        :param corpus: Corpus to train on
        :param seed: Seed of the run
        :return: Evaluation metrics of the run.
        :raises ValueError: If the corpus is empty.
        :raises ValueError: If the corpus is not annotated.
        """
        return self.train(corpus, "initial-train", seed)

    def propose_new_data(self, corpus: Corpus) -> None:
        """
        Proposes new data points and adds them as increment to the corpus
        """
        logger.info("Start new propose iteration")

        potential_ids = corpus.get_not_annotated_data_points_ids()

        sampling_budget, step_size = self.determine_step_size(len(corpus), potential_ids)

        new_data_points = self.teacher.propose(
            potential_ids, step_size, sampling_budget
        )

        detected_step_size = len(new_data_points)
        if corpus.do_i_have_to_annotate() and \
                detected_step_size != self.cfg.experiment.step_size:
            error_message = f"Step size deviation detected: " \
                            f"Actual '{detected_step_size}', expected '{self.cfg.experiment.step_size}' " \
                            f"(Corpus not exhausted!)"
            if not self.cfg.technical.adjust_wrong_step_size:
                raise ValueError(error_message)
            else:
                logger.warning(error_message)
                logger.warning(f"Take the first {self.cfg.experiment.step_size} points from the {detected_step_size}!")
                new_data_points = new_data_points[:self.cfg.experiment.step_size]

        corpus.add_increment(new_data_points)

    def determine_step_size(self, current_corpus_size: int, potential_ids: List[int]):
        step_size = min(len(potential_ids), self.cfg.experiment.step_size)
        sampling_budget = self.cfg.teacher.sampling_budget

        if len(potential_ids) < sampling_budget:
            sampling_budget = len(potential_ids)

        if current_corpus_size + step_size > self.cfg.experiment.annotation_budget:
            step_size = self.cfg.experiment.annotation_budget - current_corpus_size
            logger.info(f"Adjust step_size to {step_size} due to "
                        f"exceeding annotation_budget ({self.cfg.experiment.annotation_budget}). "
                        f"Sampling budget stays the same ({sampling_budget})")

        logger.info(f"Using step_size ({step_size}) and sampling_budget ({sampling_budget}).")
        return sampling_budget, step_size

    def test_and_log(self, corpus) -> MetricsType:
        """
        Evaluates the model on the test set and logs the tracked metrics on the propose run
        """
        test_metrics = self.trainer.evaluate()
        logger.info(f"Log test metrics to MLflow")
        for metric in self.tracking_metrics:
            MlflowClient().log_metric(
                self.parent_run_id,
                key=metric,
                value=test_metrics[metric],
                step=len(corpus)
            )

        return test_metrics
