import logging
import tempfile
from pathlib import Path

import click
import mlflow

TRAIN_FILE_CONV = "train"
TEST_FILE_CONV = "test"
DEV_FILE_CONV = "dev"

logger = logging.getLogger(__name__)


@click.command(help="Applies naming convention to data.")
@click.option(
    "--data-dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    required=True,
    help="Path to the data directory",
)
@click.option(
    "--train-file",
    type=click.Path(exists=False),
    required=False,
    help="Path to the train file in the directory, if not provided, the directory is used",
    default=TRAIN_FILE_CONV,
)
@click.option(
    "--test-file",
    type=click.Path(exists=False),
    required=False,
    help="Path to the test file in the directory, if not provided, the directory is used",
    default=TEST_FILE_CONV,
)
@click.option(
    "--dev-file",
    type=click.Path(exists=False),
    required=False,
    help="Path to the dev file in the directory, if not provided, the directory is used",
    default=DEV_FILE_CONV,
)
@click.option(
    "--file-format",
    type=str,
    required=False,
    help="Format of the given file names",
    default="jsonl",
)
def load_data(
    data_dir: str, train_file: str, test_file: str, dev_file: str, file_format: str
):
    """
    Load data from a directory.
    """
    with mlflow.start_run():
        load_local_data(data_dir, train_file, test_file, dev_file, file_format)


def load_local_data(
    data_dir: str, train_file: str, test_file: str, dev_file: str, file_format: str
) -> None:
    """
    Load data from a directory.

    Args:
        data_dir: Path to the data directory
        data_file: Path to a file in the directory, if not provided, the whole directory is used
        :param file_format: file format of the files
        :param data_dir: data directory in which the files are located
        :param train_file: train file name
        :param test_file: test file name
        :param dev_file: dev file name
    """
    data_dir_path = Path(data_dir)

    train_file_path = data_dir_path / (train_file + f".{file_format}")
    test_file_path = data_dir_path / (test_file + f".{file_format}")
    dev_file_path = data_dir_path / (dev_file + f".{file_format}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmpdir = Path(tmp_dir)
        tmpdir.joinpath(TRAIN_FILE_CONV + f".{file_format}").write_bytes(
            train_file_path.read_bytes()
        )
        tmpdir.joinpath(TEST_FILE_CONV + f".{file_format}").write_bytes(
            test_file_path.read_bytes()
        )
        tmpdir.joinpath(DEV_FILE_CONV + f".{file_format}").write_bytes(
            dev_file_path.read_bytes()
        )
        mlflow.log_artifacts(str(tmpdir), "data")


if __name__ == "__main__":
    load_data()
