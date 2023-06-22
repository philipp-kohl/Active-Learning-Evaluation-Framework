from typing import Optional, Dict

from mlflow.entities import Run
from pydantic import BaseModel

from ale.config import AppConfig
from ale.pipeline.components import PipelineComponents


class PipelineStorage(BaseModel):
    cfg: Optional[AppConfig]
    experiment_id: Optional[str]
    git_commit: Optional[str]
    completed_runs: Dict[PipelineComponents, Run] = {}

    class Config:
        arbitrary_types_allowed = True
