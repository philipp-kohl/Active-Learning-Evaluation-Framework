import logging
from pathlib import Path

import click as click
import mlflow as mlflow
import srsly
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--file",
    "-f",
    help="File for which we want to add ids",
    required=True,
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.option(
    "--output",
    "-o",
    help="Output file",
    required=True,
    type=click.Path(dir_okay=False, file_okay=True, exists=False),
)
@click.option("--start-id", help="Start ID", required=False, type=int, default=0)
def main(file: str, output: str, start_id: int):
    add_ids_to_jsonl(file, output, start_id, False)


def add_ids_to_jsonl(file: str, output: str, start_id: int, force: bool):
    file = Path(file)
    output = Path(output)

    entries_with_id = []
    for current_id, entry in tqdm(enumerate(srsly.read_jsonl(file), start_id)):
        if "id" in entry and not force:
            logger.warning("Entries seem to have an ID. Abort processing!")
            return

        entry["id"] = current_id
        entries_with_id.append(entry)

    srsly.write_jsonl(output, entries_with_id)
    mlflow.log_artifact(str(output), "data")


if __name__ == "__main__":
    main()
