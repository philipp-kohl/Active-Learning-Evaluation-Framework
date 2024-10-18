"""
Utils for usage with mlflow.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Optional, Any, List, Union

import mlflow
import pandas as pd
import srsly
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.entities import ViewType, RunStatus, Run
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def parameters_match(given_params: Dict[str, Any], params: Dict[str, Any]):
    if len(given_params) != len(params):
        logger.debug(
            f"Number of parameters do not match: {len(given_params)}:{len(params)}"
        )
        return False

    for key in given_params:
        if key not in params:
            logger.debug(f"'{key}' not in run parameters!")
            return False
        if given_params[key] != params[key]:
            logger.debug(
                f"Difference for '{key}': {given_params[key]} != {params[key]}"
            )
            return False

    return True


def tags_partially_match(given_tags: Dict[str, Any], tags: Dict[str, Any]):
    for key in given_tags:
        if key not in tags:
            logger.debug(f"'{key}' not in run tags!")
            return False

        if str(given_tags[key]) != tags[key]:
            logger.debug(f"Difference for '{key}': {given_tags[key]} != {tags[key]}")
            return False

    return True


def _already_ran(
        parameters,
        git_commit,
        experiment_id=None,
        run_name=None,
        run_status: int = RunStatus.FINISHED,
        given_tags: Optional[Dict[str, Any]] = None,
) -> Optional[Run]:
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()

    filter_string = f"attributes.status = '{RunStatus.to_string(run_status)}' "
    if run_name:
        filter_string += f"and tags.mlflow.runName = '{run_name}'"

    all_run_infos = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.end_time DESC"],
    )
    for run_info in all_run_infos:
        tags = run_info.data.tags

        if parameters is not None:
            given_params = walk_params_from_omegaconf_dict(parameters)
            if not parameters_match(given_params, run_info.data.params):
                continue

        if given_tags is not None:
            if not tags_partially_match(given_tags, tags):
                continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            logger.info(
                f"Run matched, but has a different source version, so skipping "
                f"(found={previous_version}, expected={git_commit})"
            )
            continue

        run_id = run_info.info.run_id
        logger.info(f"Matching run found for {run_name}: {run_id}")
        return client.get_run(run_id)

    logger.info("No matching run has been found.")
    return None


def flatten_dictionary(parent_name, element, global_dict: Dict[str, str]):
    if isinstance(element, DictConfig) or isinstance(element, Dict):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                flatten_dictionary(f"{parent_name}.{k}", v, global_dict)
            else:
                global_dict[f"{parent_name}.{k}"] = str(v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            global_dict[f"{parent_name}.{i}"] = str(v)
    else:
        global_dict[f"{parent_name}"] = str(element)


def walk_params_from_omegaconf_dict(
        params, function: Callable[[str, str], None] = None, error_on_long_param: bool = False,
):
    global_dict: Dict[str, str] = {}
    for param_name, element in params.items():
        flatten_dictionary(param_name, element, global_dict)

    if function is not None:
        for param_name, param_value in global_dict.items():
            to_log = param_value
            if len(str(to_log)) > 500:  # TODO log each list entry?
                if error_on_long_param:
                    raise ValueError(f"Param value '{to_log}' for '{param_name}' too long!")

                logger.warning(f"Logging to long parameter '{param_name}'")
                to_log = "TOO_LONG!"

            function(param_name, to_log)

    return global_dict


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def log_metric(run: Run, key: str, value: float, step=0):
    MlflowClient().log_metric(run.info.run_id, key, value, timestamp=int(time.time() * 1000), step=step)


def log_param(run: Run, key: str, value: Any):
    MlflowClient().log_param(run.info.run_id, key, value)


def log_artifact(run: Run, local_path: str, artifact_path: str = None):
    MlflowClient().log_artifact(run.info.run_id, local_path=local_path, artifact_path=artifact_path)


def load_artifact(run: Run, path: str, dst_path: str = None) -> str:
    return download_artifacts(run_id=run.info.run_id, artifact_path=path, dst_path=dst_path)


lock = threading.Lock()


def mark_run_as_finished(run: Run, run_status: RunStatus):
    MlflowClient().set_terminated(run.info.run_id, RunStatus.to_string(run_status))


def mark_run_as_running(run: Run):
    MlflowClient().set_terminated(run.info.run_id, RunStatus.to_string(RunStatus.RUNNING))


def log_dict_as_artifact(run: Run, value: Dict, artifact_file: str):
    with tempfile.TemporaryDirectory() as dir:
        path = Path(dir) / artifact_file
        srsly.write_json(path, value)
        log_artifact(run, str(path.resolve()))


def get_or_create_experiment(experiment_seed_name: str) -> str:
    with lock:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_seed_name)
        if experiment is None:
            return client.create_experiment(name=experiment_seed_name)
        else:
            return experiment.experiment_id


def get_all_child_runs(experiment_id: str, run_id: str, run_status: RunStatus = RunStatus.RUNNING) -> List[Run]:
    filter_string = f"attributes.status = '{RunStatus.to_string(run_status)}' " \
                    f"and tags.mlflow.parentRunId = '{run_id}'"

    client = MlflowClient()
    all_run_infos = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.end_time DESC"],
    )

    all_runs = []
    for run in all_run_infos:
        all_runs.append(run)
        child_runs = get_all_child_runs(experiment_id, run.info.run_id)
        all_runs.extend(child_runs)

    return all_runs


def store_bar_plot(distribution: Dict[str, float], mlflow_run: Run, artifact_name: str,
                   columns: List[str]) -> None:
    import plotly.express as px

    sorted_label_data = sorted(distribution.items(), key=lambda x: x[0])
    df = pd.DataFrame(sorted_label_data, columns=columns)
    fig = px.bar(df, x=columns[0], y=columns[1], title=artifact_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Define file paths within the temporary directory
        html_path = f'{temp_dir}/bar_plot.html'
        csv_path = f'{temp_dir}/data.csv'

        # Generate and save the HTML plot
        fig.write_html(html_path)
        log_artifact(mlflow_run, html_path, artifact_path=artifact_name)

        # Save the DataFrame to a CSV file
        df.to_csv(csv_path, index=False)
        log_artifact(mlflow_run, csv_path, artifact_path=artifact_name)


def store_csv(data_frame: pd.DataFrame, mlflow_run: Run, artifact_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = f'{temp_dir}/data.csv'
        # Save the DataFrame to a CSV file
        data_frame.to_csv(csv_path, index=False)
        log_artifact(mlflow_run, csv_path, artifact_path=artifact_name)


def store_histogram(data: List[Union[int, float]], mlflow_run: Run, artifact_name: str,
                    columns: List[str], bins=10) -> None:
    data = pd.DataFrame({
        columns[0]: pd.Series(data)
    })
    import plotly.express as px
    # Create a histogram using Plotly
    fig = px.histogram(data, x=columns[0], nbins=bins, title='Histogram')

    with tempfile.TemporaryDirectory() as temp_dir:
        histogram_html_path = f'{temp_dir}/histogram.html'
        fig.write_html(histogram_html_path)

        log_artifact(mlflow_run, histogram_html_path, artifact_path=artifact_name)


def store_log_file_to_mlflow(file: str, run_id: str):
    root_run = _find_root_run_id(run_id)

    new_file_name = f"{datetime.now().isoformat()}.log"

    with tempfile.TemporaryDirectory() as temp_directory:
        # Construct the full path for the new file in the temporary directory
        new_file_path = os.path.join(temp_directory, new_file_name)
        # Copy the source file to the new location with the new name
        shutil.copy(file, new_file_path)

        log_artifact(root_run, new_file_path, artifact_path="logs")


def _find_root_run_id(current_run_id: str) -> Run:
    """
    Recursively finds the root parent run ID of a given MLflow run.

    :param current_run_id: The run ID of the current MLflow run.
    :return: The run ID of the root parent run.
    """
    parent_run = mlflow.get_parent_run(current_run_id)

    # Base case: If there's no parent, this run is the root
    if not parent_run:
        return mlflow.get_run(current_run_id)
    else:
        # Recursive case: Keep looking for the parent
        return _find_root_run_id(parent_run.info.run_id)
