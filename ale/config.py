from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, model_validator


class MlFlowConfig(BaseModel):
    url: str
    experiment_name: str
    run_name: Optional[str] = None
    git_hash: Optional[str] = None
    user: Optional[str] = None
    source_name: Optional[str] = None


class TrainerConfig(BaseModel):
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


class TeacherConfig(BaseModel):
    strategy: str
    sampling_budget: int
    aggregation_method: Optional[AggregationMethod] = None


class Experiment(BaseModel):
    step_size: int
    initial_data_size: float
    """
    Value between 0 and 1 is treated as ratio.
    Value > 1 is treated as nominal value.
    """
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
    early_stopping_threshold: float
    """
    Defines the threshold for the optimization metric
    """
    early_stopping_n_iter: int
    """
    Number of consecutive iterations the threshold is reached before stopping.
    """


class TechnicalConfig(BaseModel):
    use_gpu: int
    number_threads: int
    adjust_wrong_step_size: bool


class NLPTask(str, Enum):
    CLS = "CLS"
    NER = "NER"


class DataConfig(BaseModel):
    data_dir: str
    train_file: str
    test_file: str
    dev_file: str
    file_format: str
    nlp_task: NLPTask
    text_column: Optional[str] = "text"
    label_column: Optional[str] = "label"


class ConverterConfig(BaseModel):
    converter_class: str
    target_format: str


class AppConfig(BaseModel):
    data: DataConfig
    experiment: Experiment
    mlflow: MlFlowConfig
    teacher: TeacherConfig
    trainer: TrainerConfig
    converter: ConverterConfig
    technical: TechnicalConfig

    @model_validator(mode="before")
    def check_configuration(cls, values):
        teacher: TeacherConfig = values.get("teacher")
        experiment: Experiment = values.get("experiment")

        if teacher.sampling_budget < experiment.step_size and teacher.sampling_budget != -1:
            raise ValueError(
                f"Teacher.budget ({teacher.sampling_budget}) must be >= experiment.step_size ({experiment.step_size})")

        return values
