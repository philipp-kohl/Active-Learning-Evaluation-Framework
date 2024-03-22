import logging
from pathlib import Path
from typing import List, Union, Dict, Any

import srsly

from ale.corpus.corpus import Corpus
from ale.registry.registerable_corpus import CorpusRegistry
from ale.trainer.lightning.ner_dataset import AleNerDataModule

logger = logging.getLogger(__name__)


@CorpusRegistry.register("pytorch-lightning-corpus")
class PytorchLightningCorpus(Corpus):
    def __init__(self, data_dir: Union[str, Path]):
        super().__init__(data_dir)

        def filter_relevant_ids(loaded_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            filtered = []
            for entry in loaded_data:
                if entry["id"] in self.get_relevant_ids():
                    filtered.append(entry)
            return filtered

        self.data_module = AleNerDataModule(data_dir,
                                            model_name="FacebookAI/roberta-base", # TODO
                                            labels=["PER", "ORG", "LOC", "MISC"], # TODO
                                            batch_size=32,
                                            num_workers=1,
                                            train_filter_func=filter_relevant_ids)

        logger.info("Start indexing corpus")
        self.index = {}
        for entry in srsly.read_jsonl(data_dir / "train.jsonl"):
            self.index[entry["id"]] = entry["text"]
        logger.info("End indexing corpus")

    def get_trainable_corpus(self):
        return self.data_module.train_dataloader()

    def get_not_annotated_data_points_ids(self) -> List[int]:
        all_ids = set(self.index.keys())
        already_annotated = set(self.relevant_ids)

        return list(all_ids.difference(already_annotated))

    def get_all_texts_with_ids(self) -> Dict[int, str]:
        return self.index

    def get_text_by_ids(self, idxs: List[int]) -> Dict[int, str]:
        return {idx: self.index[idx] for idx in idxs}
