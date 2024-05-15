import logging
from pathlib import Path
from typing import Optional, Dict

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


class StopAfterNAlCycles(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, "", **kwargs)
        self.iteration_counter = 1

    @override
    def on_iter_end(self) -> None:
        self.iteration_counter += 1

    def may_continue(self) -> bool:
        return self.iteration_counter <= self.cfg.experiment.stop_after_n_al_cycles





