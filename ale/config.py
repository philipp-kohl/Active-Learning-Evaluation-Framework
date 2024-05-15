from typing import Optional, List
from enum import Enum

from hydra.core.config_store import ConfigStore
from pydantic import root_validator
from pydantic.dataclasses import dataclass


@dataclass
class MlFlowConfig:
    url: str
    experiment_name: str
    run_name: Optional[str] = None
    git_hash: Optional[str] = None
    user: Optional[str] = None
    source_name: Optional[str] = None


@dataclass
class TrainerConfig:
    trainer_name: str
    huggingface_model: str
    corpus_manager: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    num_workers: int
    device: str
    early_stopping_delta: float
    early_stopping_patience: int
    label_smoothing: float
    model: str


class AggregationMethod(str, Enum):
    AVERAGE = "AVG"
    STD = "STD"
    MAXIMUM = "MAX"
    MINIMUM = "MIN"
    SUM = "SUM"


@dataclass
class TeacherConfig:
    strategy: str
    sampling_budget: int
    aggregation_method: Optional[AggregationMethod] = None


@dataclass
class Experiment:
    step_size: int
    initial_data_ratio: float
    initial_data_strategy: str
    tracking_metrics: List[str]
    seeds: List[int]
    annotation_budget: int
    assess_data_bias: bool
    assess_data_bias_eval_freq: int
    assess_overconfidence: bool
    assess_overconfidence_eval_freq: int
    stop_after_n_al_cycles: int
    """
    value < 1 is treated as no stopping.
    value > 0 is treated as stopping.
    """


@dataclass
class TechnicalConfig:
    use_gpu: int
    number_threads: int
    adjust_wrong_step_size: bool


class NLPTask(str, Enum):
    CLS = "CLS"
    NER = "NER"


@dataclass
class DataConfig:
    data_dir: str
    train_file: str
    test_file: str
    dev_file: str
    file_format: str
    nlp_task: NLPTask
    text_column: Optional[str] = "text"
    label_column: Optional[str] = "label"


@dataclass
class ConverterConfig:
    converter_class: str
    target_format: str


@dataclass
class AppConfig:
    data: DataConfig
    experiment: Experiment
    mlflow: MlFlowConfig
    teacher: TeacherConfig
    trainer: TrainerConfig
    converter: ConverterConfig
    technical: TechnicalConfig

    @root_validator(pre=True)
    def check_configuration(cls, values):
        teacher: TeacherConfig = values.get("teacher")
        experiment: Experiment = values.get("experiment")

        if teacher.sampling_budget < experiment.step_size and teacher.sampling_budget != -1:
            raise ValueError(f"Teacher.budget ({teacher.sampling_budget}) must be >= experiment.step_size ({experiment.step_size})")

        return values


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config", node=AppConfig)
