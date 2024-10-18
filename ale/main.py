from ale.import_helper import import_registrable_components
from ale.pipeline.pipeline_components.measure_data_distribution import DataDistributionMeasure

import_registrable_components()

import getpass
import logging
import os
import sys

from ale.mlflowutils.mlflow_utils import get_git_revision_hash

import hydra
from omegaconf import OmegaConf

from ale.pipeline.pipeline import MLFlowPipeline
from ale.pipeline.pipeline_components.add_ids_component import AddIdsTrainComponent, AddIdsDevComponent
from ale.pipeline.pipeline_components.aggregate_seed_runs import AggregateSeedRuns
from ale.pipeline.pipeline_components.collect_labels_component import (
    CollectLabelsComponent,
)
from ale.pipeline.components import PipelineComponents
from ale.pipeline.pipeline_components.load_data_raw_component import (
    LoadDataRawComponent,
)
from ale.pipeline.pipeline_components.load_data_converted_component import (
    LoadDataConvertedComponent,
)
from ale.pipeline.pipeline_components.propose_data_component import ProposeDataComponent
from ale.registry.registerable_pipeline_component import PipelineComponentRegistry
from config import AppConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def ale(cfg: AppConfig) -> None:
    """
    🍻
    """
    OmegaConf.to_object(cfg)
    logging.info(str(cfg))
    run(cfg)


def run(cfg: AppConfig):
    """
    Start the experiment using the provided configuration and mlflow.

    :param cfg: The configuration to use.
    """
    if "MLFLOW_TRACKING_URI" in os.environ:
        logger.warning(f"MLFLOW_TRACKING_URI set via environment variable ({os.environ['MLFLOW_TRACKING_URI']}). "
                       f"It will be replaced with '{cfg.mlflow.url}' for this process!")

    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.url
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(cfg.mlflow.max_retries)
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(cfg.mlflow.timeout)
    os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = str(cfg.mlflow.backoff_factor)
    os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_JITTER"] = str(cfg.mlflow.backoff_jitter)

    cfg.mlflow.user = getpass.getuser()
    cfg.mlflow.source_name = sys.argv[0]
    cfg.mlflow.git_hash = get_git_revision_hash()

    converter_class = PipelineComponentRegistry.get_instance(
        cfg.converter.converter_class
    )

    pipeline = MLFlowPipeline(cfg)
    pipeline.add(PipelineComponents.ADD_IDS_TO_TRAIN_FILE, AddIdsTrainComponent)
    pipeline.add(PipelineComponents.ADD_IDS_TO_DEV_FILE, AddIdsDevComponent)
    pipeline.add(PipelineComponents.COLLECT_LABELS, CollectLabelsComponent)
    pipeline.add(PipelineComponents.DATA_DISTRIBUTIONS, DataDistributionMeasure)
    pipeline.add(PipelineComponents.CONVERT_DATA, converter_class)
    pipeline.add(PipelineComponents.LOAD_DATA_RUN_RAW, LoadDataRawComponent)
    pipeline.add(PipelineComponents.LOAD_DATA_RUN_CONVERTED, LoadDataConvertedComponent)
    pipeline.add(PipelineComponents.SEED_RUNS, ProposeDataComponent)
    pipeline.add(PipelineComponents.AGGREGATE_SEED_RUNS, AggregateSeedRuns)
    pipeline.start()


if __name__ == "__main__":
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/#disable-changing-current-working-dir-to-jobs-output-dir
    # torch.use_deterministic_algorithms(True, warn_only=True)
    sys.argv.append("hydra.job.chdir=False")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ale()
