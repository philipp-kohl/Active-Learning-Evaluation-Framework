import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import List, Any, Dict

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus, Run
from mlflow.utils import mlflow_tags

import ale.mlflowutils.mlflow_utils as utils
from ale.config import AppConfig
from ale.proposer.proposer_per_seed import AleBartenderPerSeed

logger = logging.getLogger(__name__)


class AleBartender:
    def __init__(self, converted_data: Path, raw_data: Path, cfg: AppConfig, labels: List[Any]):
        self.cfg = cfg
        self.labels = labels
        self.converted_data_dir = converted_data
        self.train_file_raw = raw_data / "train.jsonl"
        self.dev_file_raw = raw_data / "dev.jsonl"

        self.seeds = self.cfg.experiment.seeds

        logger.info(
            f"Tracking following metrics: {self.cfg.experiment.tracking_metrics} to {self.cfg.mlflow.url}"
        )
        self.tracking_metrics: List[str] = self.cfg.experiment.tracking_metrics

        self.run_id = mlflow.active_run().info.run_id

    def run(self) -> None:
        """
        Runs the experiment for every seed.
        Setup instances of corpus, trainer and teacher using the classes provided by each registry.
        """
        number_threads = self.cfg.technical.number_threads
        if number_threads == -1:
            number_threads = len(self.seeds)

        logger.info(f"Starting thread pool with {number_threads} threads to run {len(self.seeds)} seeds.")
        executor = ThreadPoolExecutor(max_workers=number_threads)
        run_ids = [run_id for run_id in executor.map(self.resume_or_start_seed_run, self.seeds)]
        executor.shutdown(wait=True)

        logger.info(f"All seed runs finished. Run ids to aggregate: {run_ids}")


    def resume_or_start_seed_run(self, seed) -> str:
        experiment_seed_id = utils.get_or_create_experiment(self.cfg.mlflow.experiment_name)
        run_name = f"{self.cfg.teacher.strategy}-with-seed-{seed}"
        seed_tag = {"seed": str(seed)}
        logger.debug(f"Seed tag: {seed_tag}")
        matching_run = utils._already_ran(
            self.cfg,
            utils.get_git_revision_hash(),
            experiment_id=experiment_seed_id,
            run_status=RunStatus.FINISHED,
            run_name=run_name,
            given_tags=seed_tag,
        )
        if matching_run is None:
            logger.warning(
                "No finished run found. Try to find a failed run to resume."
            )

            matching_run = utils._already_ran(
                self.cfg,
                utils.get_git_revision_hash(),
                experiment_id=experiment_seed_id,
                run_status=RunStatus.FAILED,
                run_name=run_name,
                given_tags=seed_tag,
            )

            if matching_run is None:
                matching_run = utils._already_ran(
                    self.cfg,
                    utils.get_git_revision_hash(),
                    experiment_id=experiment_seed_id,
                    run_status=RunStatus.RUNNING,
                    run_name=run_name,
                    given_tags=seed_tag,
                )

                if matching_run is None:
                    run = self.start_new_seed_run(experiment_seed_id, run_name, seed, seed_tag)
                    return run.info.run_id
                else:
                    logger.warning(f"Found running run ({matching_run.info.run_id}) and resume it!")
                    run = self.resume_seed_run(matching_run, seed)
                    return run.info.run_id
            else:
                run = self.resume_seed_run(matching_run, seed)
                return run.info.run_id

    def resume_seed_run(self, run: Run, seed: int):
        logger.info(f"Resume seed run: {seed}")
        try:
            utils.mark_run_as_running(run)
            seed_simulator = AleBartenderPerSeed(self.cfg,
                                                 seed,
                                                 self.converted_data_dir,
                                                 self.train_file_raw,
                                                 self.dev_file_raw,
                                                 self.labels,
                                                 run.info.experiment_id,
                                                 run.info.run_id,
                                                 self.tracking_metrics)
            seed_simulator.run_single_seed()
            utils.mark_run_as_finished(run, RunStatus.FINISHED)
        except Exception as e:
            utils.mark_run_as_finished(run, RunStatus.FAILED)
            raise e
        finally:
            utils.store_log_file_to_mlflow("main.log", run.info.run_id)

        return run


    def start_new_seed_run(self, experiment_seed_id: str, run_name: str, seed: int, seed_tag: Dict[str, Any]):
        logger.info(f"Start new seed run: {seed}")
        logger.info(f"Start child run: {run_name}")
        client = MlflowClient()
        tags = seed_tag.copy()
        tags[mlflow_tags.MLFLOW_PARENT_RUN_ID] = self.run_id
        tags[mlflow_tags.MLFLOW_GIT_COMMIT] = self.cfg.mlflow.git_hash
        tags[mlflow_tags.MLFLOW_USER] = self.cfg.mlflow.user
        tags[mlflow_tags.MLFLOW_SOURCE_NAME] = self.cfg.mlflow.source_name
        tags[mlflow_tags.MLFLOW_SOURCE_TYPE] = "LOCAL"
        run = client.create_run(
            experiment_id=experiment_seed_id,
            run_name=run_name,
            tags=tags)
        try:
            utils.walk_params_from_omegaconf_dict(
                self.cfg, lambda key, value: utils.log_param(run, key, value)
            )
            seed_simulator = AleBartenderPerSeed(self.cfg,
                                                 seed,
                                                 self.converted_data_dir,
                                                 self.train_file_raw,
                                                 self.dev_file_raw,
                                                 self.labels,
                                                 experiment_seed_id,
                                                 run.info.run_id,
                                                 self.tracking_metrics)
            seed_simulator.run_single_seed()
            utils.mark_run_as_finished(run, RunStatus.FINISHED)
        except Exception as e:
            utils.mark_run_as_finished(run, RunStatus.FAILED)
            raise e
        finally:
            utils.store_log_file_to_mlflow("main.log", run.info.run_id)
        return run
