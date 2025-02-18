import logging
from pathlib import Path
import random
from typing import Any, List, Tuple, Optional

import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.entities import RunStatus, Run
from mlflow.utils import mlflow_tags
from torch.utils.data import DataLoader

import ale.mlflowutils.mlflow_utils as utils
from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.proposer.hooks.abstract_hook import ProposeHook
from ale.proposer.hooks.assess_bias_hook import AssessBiasHook
from ale.proposer.hooks.assess_confidence_hook import AssessConfidenceHook
from ale.proposer.hooks.early_stopping import EarlyStopping
from ale.proposer.hooks.measure_times import MeasureTimes
from ale.proposer.hooks.stop_after_n_al_cycles import StopAfterNAlCycles
from ale.registry.registerable_corpus import CorpusRegistry
from ale.registry.registerable_teacher import TeacherRegistry
from ale.registry.registerable_trainer import TrainerRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.base_trainer import MetricsType
from ale.trainer.prediction_trainer import PredictionTrainer

logger = logging.getLogger(__name__)


class AleBartenderPerSeed:

    def __init__(self,
                 cfg: AppConfig,
                 seed: int,
                 converted_data_dir: Path,
                 train_file_raw: Path,
                 dev_file_raw: Path,
                 labels: List[Any],
                 experiment_id: str,
                 parent_run_id: str,
                 tracking_metrics: List[str]):
        self.cfg = cfg
        self.seed = self.seed_everything(seed=seed)
        self.experiment_id = experiment_id
        self.parent_run_id = parent_run_id
        self.tracking_metrics = tracking_metrics

        logger.info(f"Use corpus manager: {self.cfg.trainer.corpus_manager}")
        corpus_class = CorpusRegistry.get_instance(self.cfg.trainer.corpus_manager)
        self.corpus = corpus_class(cfg, converted_data_dir, labels)
        logger.info(f"Use trainer: {self.cfg.trainer.trainer_name}")
        trainer_class = TrainerRegistry.get_instance(
            self.cfg.trainer.trainer_name
        )
        self.trainer: PredictionTrainer = trainer_class(
            self.cfg,
            self.corpus,
            # Path(self.cfg.trainer.config_path),
            # self.cfg.technical.use_gpu,
            self.seed,
            labels,
            # self.cfg.data.nlp_task,
            # self.cfg.trainer.recreate_pipeline_each_run
        )
        logger.info(f"Use strategy: {self.cfg.teacher.strategy}")
        teacher_strategy_class = TeacherRegistry.get_instance(
            self.cfg.teacher.strategy
        )

        if "aggregation_method" in self.cfg.teacher and self.cfg.teacher.aggregation_method:
            self.teacher: BaseTeacher = teacher_strategy_class(
                corpus=self.corpus,
                predictor=self.trainer,
                seed=self.seed,
                labels=labels,
                nlp_task=self.cfg.data.nlp_task,
                aggregation_method=self.cfg.teacher.aggregation_method)
        else:
            self.teacher: BaseTeacher = teacher_strategy_class(
                corpus=self.corpus,
                predictor=self.trainer,
                seed=self.seed,
                labels=labels,
                nlp_task=self.cfg.data.nlp_task)

        logger.info(f"Use '{self.cfg.experiment.initial_data_strategy}' teacher for initial data ratio.")
        initial_teacher_strategy_class = TeacherRegistry.get_instance(
            self.cfg.experiment.initial_data_strategy
        )
        self.initial_teacher: BaseTeacher = initial_teacher_strategy_class(
            corpus=self.corpus,
            predictor=self.trainer,
            seed=self.seed,
            labels=labels,
            nlp_task=self.cfg.data.nlp_task
        )
        self.dev_file_raw = dev_file_raw
        self.train_file_raw = train_file_raw
        # self.train_file_converted = train_file_converted

    def run_single_seed(self) -> None:
        """
        Run the experiment for a single seed. Setup the internal random state with the given seed.
        Starts the initial training on the first increment of not annotated data points,
        then continues with the teacher-proposing and training process.
        :param seed: Seed to run the experiment with.
        :return None
        """
        logger.info(f"Start seed: {self.seed}")

        all_ids = self.corpus.get_not_annotated_data_points_ids()
        first_step_size = self.determine_initial_step_size(all_ids)

        logger.info(f"Initial step size: {first_step_size}")

        old_run: Optional[Run] = None
        already_runned_runs = utils.get_all_child_runs(self.experiment_id, self.parent_run_id, RunStatus.FINISHED)
        if already_runned_runs:
            newest_run = already_runned_runs[0]
            logger.info(f"Restore run: {newest_run.info.run_name}")
            self.trainer.restore_from_artifacts(newest_run)
            self.corpus.restore_from_artifacts(newest_run)
            self.teacher.restore_from_artifacts(newest_run)
            old_run = newest_run
        else:
            initial_data_ids = self.initial_teacher.propose(all_ids, first_step_size, self.cfg.teacher.sampling_budget)
            self.corpus.add_increment(initial_data_ids)
            initial_train_evaluation_metrics, _, run = self.initial_train(self.corpus, self.seed)
            old_run = run
            self.teacher.after_initial_train(initial_train_evaluation_metrics)
            utils.store_log_file_to_mlflow("main.log", run.info.run_id)

        annotation_budget: int = self.cfg.experiment.annotation_budget

        # The MeasureTimes hook should always be the first in the list. Thus, we do not measure much overhead.
        hooks: List[ProposeHook] = [MeasureTimes(self.cfg, self.parent_run_id, self.corpus)]
        if self.cfg.experiment.stop_after_n_al_cycles > 0:
            hooks.append(StopAfterNAlCycles(self.cfg, self.parent_run_id, self.corpus))
        if self.cfg.experiment.early_stopping_threshold > 0:
            hooks.append(EarlyStopping(self.cfg, self.parent_run_id, self.corpus))
        if self.cfg.experiment.assess_data_bias:
            hooks.append(AssessBiasHook(self.cfg, self.parent_run_id, self.corpus,
                                        train_file_raw=self.train_file_raw,
                                        dev_file_raw=self.dev_file_raw,
                                        trainer=self.trainer))
        if self.cfg.experiment.assess_overconfidence:
            hooks.append(AssessConfidenceHook(self.cfg, self.parent_run_id, self.corpus, trainer=self.trainer))

        while self.corpus.do_i_have_to_annotate():
            [h.on_iter_start() for h in hooks]
            if len(self.corpus) >= annotation_budget:
                logger.info(f"Stop seed run due to exceeded annotation budget ({annotation_budget})")
                break

            if self.may_continue(hooks) is False:
                break

            [h.before_proposing() for h in hooks]
            self.propose_new_data(self.corpus)
            [h.after_proposing() for h in hooks]

            [h.before_training() for h in hooks]
            evaluation_metrics, test_metrics, new_run = self.train(
                self.corpus, f"train {len(self.corpus)}", self.seed
            )
            self.teacher.after_train(evaluation_metrics)
            self.teacher.store_state(new_run)
            [h.after_training(new_run, evaluation_metrics, test_metrics) for h in hooks]

            [h.before_prediction() for h in hooks]
            preds_train = None
            preds_dev = None
            if any([h.needs_train_predictions() for h in hooks]):
                logger.info(f"Perform predictions on training data")
                preds_train = self.perform_predictions(self.corpus.data_module.train_dataloader())
            if any([h.needs_dev_predictions() for h in hooks]):
                logger.info(f"Perform predictions on dev data")
                preds_dev = self.perform_predictions(self.corpus.data_module.val_dataloader())

            [h.after_prediction(new_run, preds_train, preds_dev) for h in hooks]

            # Delete artifacts for old run. We do not need them for resume
            self.trainer.delete_artifacts(old_run)
            old_run = new_run
            [h.on_iter_end() for h in hooks]
            logger.info("Iteration end. Store log file!")
            utils.store_log_file_to_mlflow("main.log", new_run.info.run_id)

        [h.on_seed_end() for h in hooks]
        logger.info("End seed: %s", self.seed)

    def determine_initial_step_size(self, all_ids: List[int]):
        initial_data_parameter = self.cfg.experiment.initial_data_size
        if 0.0 <= initial_data_parameter <= 1.00:
            logger.info(f"Initial step size treated as ratio ({initial_data_parameter})")
            initial_data_ratio = initial_data_parameter
            first_step_size = int(initial_data_ratio * len(all_ids))
        else:
            logger.info(f"Initial step size treated as absolute value ({initial_data_parameter})")
            first_step_size = int(initial_data_parameter)
        return first_step_size

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

        new_data_points = self.teacher.propose(potential_ids, step_size, sampling_budget)

        detected_step_size = len(new_data_points)
        if corpus.do_i_have_to_annotate() and \
                detected_step_size != step_size:
            error_message = f"Step size deviation detected: " \
                            f"Actual '{detected_step_size}', expected '{step_size}' " \
                            f"(Corpus not exhausted!)"
            if not self.cfg.technical.adjust_wrong_step_size:
                raise ValueError(error_message)
            else:
                logger.warning(error_message)
                if detected_step_size > step_size:
                    logger.warning(f"Take the first {step_size} points from the {detected_step_size}!")
                    new_data_points = new_data_points[:step_size]
                else:
                    logger.warning(
                        f"Too few datapoints selected: {detected_step_size}/{step_size}! Adding random ones.")
                    remaining_pot_ids = [idx for idx in potential_ids if idx not in new_data_points]
                    number_of_data_points_needed = step_size - detected_step_size
                    additional_data_points = random.sample(remaining_pot_ids, number_of_data_points_needed)
                    new_data_points.extend(additional_data_points)

        corpus.add_increment(new_data_points)

    def determine_step_size(self, current_corpus_size: int, potential_ids: List[int]):
        step_size = min(len(potential_ids), self.cfg.experiment.step_size)
        sampling_budget = self.cfg.teacher.sampling_budget

        if len(potential_ids) <= sampling_budget or sampling_budget == -1:
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

    def perform_predictions(self, data_loader: DataLoader):
        return self.trainer.predict_with_known_gold_labels(data_loader)

    def may_continue(self, hooks: List[ProposeHook]) -> bool:
        for hook in hooks:
            if hook.may_continue() is False:
                logger.info(f"Hook ({hook.__class__.__name__}) caused the AL cycle to stop!")
                return False
        return True

    def seed_everything(self, seed: int) -> int:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        if not (min_seed_value <= seed <= max_seed_value):
            adjusted_seed = self.adjust_seed(seed, max_seed_value)
            logger.warning(f"Seed '{seed}' is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}. "
                           f"ALE adjust seed to {adjusted_seed}.")
            seed = adjusted_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        return seed

    def adjust_seed(self, number: int, max_value: int):
        # Get the number of digits in the max_value
        max_digits = len(str(max_value))

        # Convert the number to string and trim its last digits to match the max_digits length
        trimmed_number_str = str(number)[:max_digits]

        # Convert back to integer
        trimmed_number = int(trimmed_number_str)

        # Ensure the trimmed number does not exceed the max_value
        if trimmed_number > max_value:
            trimmed_number = int(trimmed_number_str[:max_digits - 1])

        return trimmed_number

