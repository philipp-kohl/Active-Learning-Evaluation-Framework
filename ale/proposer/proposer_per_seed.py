import logging
import random
import time
from pathlib import Path
from typing import Any, List, Tuple, Optional

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus, Run
from mlflow.utils import mlflow_tags

import ale.mlflowutils.mlflow_utils as utils
from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.registry.registerable_corpus import CorpusRegistry
from ale.registry.registerable_teacher import TeacherRegistry
from ale.registry.registerable_trainer import TrainerRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.base_trainer import MetricsType

logger = logging.getLogger(__name__)


class AleBartenderPerSeed:

    def __init__(self,
                 cfg: AppConfig,
                 seed: int,
                 train_file_converted: Path,
                 dev_file_converted: Path,
                 test_file_converted: Path,
                 labels: List[Any],
                 experiment_id: str,
                 parent_run_id: str,
                 tracking_metrics: List[str]):
        self.cfg = cfg
        self.seed = seed
        self.experiment_id = experiment_id
        self.parent_run_id = parent_run_id
        self.tracking_metrics = tracking_metrics

        logger.info(f"Use corpus mananger: {self.cfg.trainer.corpus_manager}")
        corpus_class = CorpusRegistry.get_instance(self.cfg.trainer.corpus_manager)
        self.corpus = corpus_class(train_file_converted, self.cfg.trainer)
        logger.info(f"Use trainer: {self.cfg.trainer.trainer_name}")
        trainer_class = TrainerRegistry.get_instance(
            self.cfg.trainer.trainer_name
        )
        self.trainer = trainer_class(
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
            initial_data_ids = self.initial_teacher.propose(all_ids, first_step_size, self.cfg.teacher.budget)
            self.corpus.add_increment(initial_data_ids)
            initial_train_evaluation_metrics, run = self.initial_train(self.corpus, self.seed)
            old_run = run
            self.teacher.after_initial_train(initial_train_evaluation_metrics)

        while self.corpus.do_i_have_to_annotate():
            self.propose_new_data(self.corpus)

            evaluation_metrics, new_run = self.train(
                self.corpus, f"train {len(self.corpus)}", self.seed
            )
            # Delete artifacts for old run. We do not need them for resume
            self.trainer.delete_artifacts(old_run)
            old_run = new_run
            self.teacher.after_train(evaluation_metrics)
        logger.info("End seed: %s", self.seed)

    def train(self, corpus: Corpus, run_name: str, seed: int) -> Tuple[MetricsType, Run]:
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
            self.test_and_log(corpus)
            self.trainer.store_to_artifacts(train_run)
            corpus.store_to_artifacts(train_run)
            utils.mark_run_as_finished(train_run, RunStatus.FINISHED)

            return evaluation_metrics, train_run
        except Exception as e:
            utils.mark_run_as_finished(train_run, RunStatus.FAILED)
            raise e

    def initial_train(self, corpus: Corpus, seed: int) -> Tuple[MetricsType, Run]:
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

        step_size = min(len(potential_ids), self.cfg.experiment.step_size)
        budget = self.cfg.teacher.budget
        if len(potential_ids) < budget:
            budget = len(potential_ids)

        logger.info(f"Using step_size ({step_size}) and budget ({budget}).")
        new_data_points = self.teacher.propose(
            potential_ids, step_size, budget
        )

        detected_step_size = len(new_data_points)
        if corpus.do_i_have_to_annotate() and \
                detected_step_size != self.cfg.experiment.step_size:
            error_message = f"Step size deviation detected: " \
                            f"Actual '{detected_step_size}', expected '{self.cfg.experiment.step_size}' "\
                            f"(Corpus not exhausted!)"
            if not self.cfg.technical.adjust_wrong_step_size:
                raise ValueError(error_message)
            else:
                logger.warning(error_message)
                logger.warning(f"Take the first {self.cfg.experiment.step_size} points from the {detected_step_size}!")
                new_data_points = new_data_points[:self.cfg.experiment.step_size]

        corpus.add_increment(new_data_points)

    def test_and_log(self, corpus) -> None:
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
