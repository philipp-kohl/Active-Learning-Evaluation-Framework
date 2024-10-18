import logging
import time
from typing import Optional, Dict

from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.proposer.hooks.abstract_hook import ProposeHook
from ale.trainer.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


class MeasureTimes(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, "", **kwargs)
        self.before_times = {}
        self.after_times = {}

    def _log_time(self, event: str, before: bool):
        if before:
            self.before_times[event] = time.time()
        else:
            self.after_times[event] = time.time()
            duration = self.after_times[event] - self.before_times[event]
            MlflowClient().log_metric(
                self.parent_run_id,
                key=f"{event}_duration",
                value=duration,
                step=len(self.corpus)
            )

    @override
    def before_proposing(self) -> None:
        self._log_time('proposing', before=True)

    @override
    def after_proposing(self) -> None:
        self._log_time('proposing', before=False)

    @override
    def before_training(self) -> None:
        self._log_time('training', before=True)

    @override
    def after_training(self, mlflow_run: Run, dev_metrics, test_metrics) -> None:
        self._log_time('training', before=False)

    @override
    def on_iter_start(self) -> None:
        self._log_time('iteration', before=True)

    @override
    def on_iter_end(self) -> None:
        self._log_time('iteration', before=False)

    @override
    def before_prediction(self) -> None:
        self._log_time('prediction', before=True)

    @override
    def after_prediction(self,
                         mlflow_run: Run,
                         preds_train: Optional[Dict[int, PredictionResult]],
                         preds_dev: Optional[Dict[int, PredictionResult]]) -> None:
        self._log_time('prediction', before=False)
