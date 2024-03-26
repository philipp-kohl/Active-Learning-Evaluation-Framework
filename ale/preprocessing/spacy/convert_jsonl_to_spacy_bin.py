import logging
import tempfile
from pathlib import Path
from typing import List

import click
import mlflow as mlflow
import spacy
import srsly
from spacy.tokens import Doc
from spacy.tokens import DocBin
from tqdm import tqdm

from ale.config import NLPTask
from ale.trainer import SpacyOnlineTrainer

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
    multiple=True,
)
@click.option("--text-column", type=str, default="text")
@click.option("--label-column", type=str, default="label")
@click.option("--id-column", type=str, default="id")
@click.option("--label", multiple=True, required=True)
@click.option("--language", type=str, default="en")
@click.option("--force", "-f", is_flag=True, default=False)
def main(
    input_path: List[str],
    text_column: str,
    label_column: str,
    id_column: str,
    label: List[str],
    force: bool,
    language: str,
    nlp_task: NLPTask = NLPTask.CLS,
):
    convert_json_to_spacy_doc_bin(
        input_path,
        text_column,
        label_column,
        id_column,
        label,
        force,
        language,
        nlp_task,
    )


def convert_json_to_spacy_doc_bin(
    input_path: List[str],
    text_column: str,
    label_column: str,
    id_column: str,
    label: List[str],
    force: bool,
    language: str,
    nlp_task: NLPTask,
):
    Doc.set_extension("active_learning_id", default=-1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in input_path:
            convert_single_file(
                force,
                file,
                label,
                label_column,
                language,
                tmp_dir,
                text_column,
                id_column,
                nlp_task,
            )

        mlflow.log_artifacts(str(tmp_dir), "data")


def convert_single_file(
    force: bool,
    input_path: str,
    label: List[str],
    label_column: str,
    language: str,
    output_path: str,
    text_column: str,
    id_column: str,
    nlp_task: NLPTask,
):
    input_path: Path = Path(input_path)
    output_path: Path = Path(output_path)
    all_labels = label
    doc_bin = DocBin(store_user_data=True)
    nlp = spacy.blank(language)

    output_file = output_path / f"{input_path.stem}.spacy"

    if output_file.exists() and force is not True:
        raise Exception(
            f"File '{output_file}' already exists. Use -f if you want to overwrite the file."
        )

    if not output_path.exists():
        output_path.mkdir(parents=True)
    for entry in tqdm(srsly.read_jsonl(input_path)):
        doc = create_doc(
            all_labels, entry, label_column, nlp, text_column, id_column, nlp_task
        )
        doc_bin.add(doc)

    doc_bin.to_disk(output_file)
    logger.info(f"File stored at {output_file}")


def add_cats(all_labels, current_label, doc):
    doc.cats = {key: 0.0 for key in all_labels}
    doc.cats[current_label] = 1.0


def add_ents(all_labels, current_label, doc):
    all_spans = []
    for start, end, label in current_label:
        current_span = doc.char_span(start, end, label=label, alignment_mode="strict")

        if current_span is None:
            current_span = doc.char_span(
                start, end, label=label, alignment_mode="contract"
            )

        if current_span is None:
            current_span = doc.char_span(
                start, end, label=label, alignment_mode="expand"
            )

        if current_span is None:
            logger.warning(f"Ignoring label '{start, end, label}'")
        else:
            all_spans.append(current_span)

    doc.spans[SpacyOnlineTrainer.SPANCAT_KEY] = all_spans


STRATEGY = {NLPTask.CLS: add_cats, NLPTask.NER: add_ents}


def create_doc(all_labels, entry, label_column, nlp, text_column, id_column, nlp_task):
    text = entry[text_column]
    current_label = entry[label_column]
    doc = nlp(text)

    STRATEGY[nlp_task](all_labels, current_label, doc)

    if id_column in entry:
        doc._.active_learning_id = entry[id_column]
    else:
        pass
        # print("No id column!")

    return doc


if __name__ == "__main__":
    main()
