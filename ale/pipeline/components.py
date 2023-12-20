from enum import Enum


class PipelineComponents(Enum):
    ADD_IDS_TO_TRAIN_FILE = "add_ids_to_train_file"
    COLLECT_LABELS = "collect_labels"
    DATA_DISTRIBUTIONS = "measure_data_distributions"
    CONVERT_DATA = "convert_data"
    LOAD_DATA_RUN_RAW = "load_data_run_raw"
    LOAD_DATA_RUN_CONVERTED = "load_data_run_converted"
    SEED_RUNS = "seed_runs"
    AGGREGATE_SEED_RUNS = "agg_seed_runs"
