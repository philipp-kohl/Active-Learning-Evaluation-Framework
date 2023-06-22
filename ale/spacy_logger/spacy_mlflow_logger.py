import sys
from types import ModuleType
from typing import IO, Tuple, Callable, Optional, Dict, Any, Iterator, List

import mlflow
import spacy
from mlflow import MlflowClient
from spacy import Language, load
from spacy.training.loggers import console_logger


@spacy.registry.loggers("ale.mlflow_logger.v1")
def mlflow_logger(
        run_id: str
    ):
    def setup_logger(
        nlp: Language = spacy.blank("en"),
        stdout: IO = sys.stdout,
        stderr: IO = sys.stderr,
    ) -> Tuple[Callable, Callable]:
        console_log, finalize_log = console_logger()(nlp)

        def log_step(info: Optional[Dict[str, Any]]):
            console_log(info)
            if info:
                _log_step_mlflow(mlflow, info, run_id)

        def finalize():
            finalize_log()
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

        return log_step, finalize

    return setup_logger


def _log_step_mlflow(
    mlflow: ModuleType,
    info: Optional[Dict[str, Any]],
    run_id: str
):
    if info is None:
        return

    client = MlflowClient()
    score = info["score"]
    other_scores = info["other_scores"]
    losses = info["losses"]
    output_path = info.get("output_path", None)
    if score is not None:
        client.log_metric(run_id, "score", score)
    if losses:
        for k, v in losses.items():
            client.log_metric(run_id, f"loss_{k}", v)
    if isinstance(other_scores, dict):
        for k, v in dict_to_dot(other_scores).items():
            if isinstance(v, float) or isinstance(v, int):
                client.log_metric(run_id, k, v)
    if output_path and score == max(info["checkpoints"])[0]:
        nlp = load(output_path)
        #mlflow.spacy.log_model(nlp, "best") # TODO


def walk_dict(
    node: Dict[str, Any], parent: List[str] = []
) -> Iterator[Tuple[List[str], Any]]:
    """Walk a dict and yield the path and values of the leaves."""
    for key, value in node.items():
        key_parent = [*parent, key]
        if isinstance(value, dict):
            yield from walk_dict(value, key_parent)
        else:
            yield (key_parent, value)


def dict_to_dot(obj: Dict[str, dict]) -> Dict[str, Any]:
    """Convert dot notation to a dict. For example: {"token": {"pos": True,
    "_": {"xyz": True }}} becomes {"token.pos": True, "token._.xyz": True}.
    values (Dict[str, dict]): The dict to convert.
    RETURNS (Dict[str, Any]): The key/value pairs.
    """
    return {".".join(key): value for key, value in walk_dict(obj)}
