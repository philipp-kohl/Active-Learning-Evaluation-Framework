import logging

from typing_extensions import override

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.proposer.hooks.abstract_hook import ProposeHook

logger = logging.getLogger(__name__)


class StopAfterNAlCycles(ProposeHook):
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        super().__init__(cfg, parent_run_id, corpus, "", **kwargs)
        self.iteration_counter = 1

    @override
    def on_iter_end(self) -> None:
        self.iteration_counter += 1

    def may_continue(self) -> bool:
        if self.iteration_counter <= self.cfg.experiment.stop_after_n_al_cycles:
            return True
        else:
            logger.warning(f"Stopping AL cycle after {self.iteration_counter} iterations")
            return False





