import logging
from typing import Dict

from mlflow.entities import Run
from typing_extensions import override

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.proposer.hooks.abstract_hook import ProposeHook

logger = logging.getLogger(__name__)


class EarlyStopping(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, "", **kwargs)
        self.threshold_reached_over_n_iter = 0

    @override
    def after_training(self, mlflow_run: Run, dev_metrics, test_metrics: Dict[str, float]) -> None:
        tracking_metric = self.cfg.experiment.tracking_metrics[0]
        metric_value = test_metrics[tracking_metric]
        if metric_value < self.cfg.experiment.early_stopping_threshold:
            logger.info("Reset early stopping iteration counter")
            self.threshold_reached_over_n_iter = 0
        else:
            self.threshold_reached_over_n_iter += 1
            logger.info(
                f"Increased early stopping iteration counter "
                f"({self.threshold_reached_over_n_iter}/{self.cfg.experiment.early_stopping_n_iter})")

    def may_continue(self) -> bool:
        if self.threshold_reached_over_n_iter < self.cfg.experiment.early_stopping_n_iter:
            return True
        else:
            logger.warning(f"Early stopping triggered at threshold "
                           f"({self.cfg.experiment.early_stopping_threshold}) "
                           f"over {self.threshold_reached_over_n_iter} iterations.")
            return False
