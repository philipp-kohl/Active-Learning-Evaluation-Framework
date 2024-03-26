import spacy
import srsly
from datasets import load_dataset
from spacy.tokens import Doc
from spacy.training import biluo_tags_to_spans, iob_to_biluo

conll2003 = load_dataset("conll2003")
train = conll2003["train"]
dev = conll2003["validation"]
test = conll2003["test"]

nlp = spacy.blank("en")


def save_to_jsonl(dataset, output_name):
    result = []
    for example in dataset:

        def replace_tag(tag_number: int):
            return dataset.features["ner_tags"].feature.names[tag_number]

        datapoint = {
            "tokens": example["tokens"],
            "text": " ".join(example["tokens"]),
            "tags": [replace_tag(tag) for tag in example["ner_tags"]],
        }

        words = example["tokens"]
        spaces = len(words) * [True]
        doc = Doc(nlp.vocab, words=words, spaces=spaces)

        tmp = biluo_tags_to_spans(doc, iob_to_biluo(datapoint["tags"]))

        char_spans = [(span.start_char, span.end_char, span.label_) for span in tmp]

        datapoint["labels"] = char_spans
        result.append(datapoint)

    srsly.write_jsonl(output_name, result)


save_to_jsonl(train, "conll2003_train.jsonl")
save_to_jsonl(dev, "conll2003_dev.jsonl")
save_to_jsonl(test, "conll2003_test.jsonl")
