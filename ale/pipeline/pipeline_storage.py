from typing import Optional, Dict

from mlflow.entities import Run
from pydantic import BaseModel

from ale.config import AppConfig
from ale.pipeline.components import PipelineComponents


class PipelineStorage(BaseModel):
    cfg: Optional[AppConfig] = None
    experiment_id: Optional[str] = None
    git_commit: Optional[str] = None
    completed_runs: Dict[PipelineComponents, Run] = {}

    class Config:
        arbitrary_types_allowed = True
