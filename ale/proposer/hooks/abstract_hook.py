from mlflow import ActiveRun
from mlflow.entities import Run

from ale.config import AppConfig
from ale.corpus.corpus import Corpus


class ProposeHook:
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        self.cfg = cfg
        self.parent_run_id = parent_run_id
        self.corpus = corpus
        self.kwargs = kwargs

    def before_proposing(self) -> None:
        pass

    def after_proposing(self) -> None:
        pass

    def after_training(self, mlflow_run: Run) -> None:
        pass

    def on_iter_end(self) -> None:
        pass

    def on_seed_end(self) -> None:
        pass
