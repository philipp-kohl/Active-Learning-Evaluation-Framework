import logging
from pathlib import Path
from typing import List, Union, Dict, Any

import srsly

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.registry.registerable_corpus import CorpusRegistry
from ale.trainer.lightning.ner_dataset import AleNerDataModule

logger = logging.getLogger(__name__)


@CorpusRegistry.register("pytorch-lightning-corpus")
class PytorchLightningCorpus(Corpus):
    def __init__(self, cfg: AppConfig, data_dir: Union[str, Path], labels: List[str]):
        super().__init__(cfg, data_dir)

        def filter_relevant_ids(loaded_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            filtered = []
            for entry in loaded_data:
                if entry["id"] in self.get_relevant_ids():
                    filtered.append(entry)
            return filtered

        self.data_module = AleNerDataModule(data_dir,
                                            model_name=self.cfg.trainer.huggingface_model,
                                            labels=labels,
                                            batch_size=self.cfg.trainer.batch_size,
                                            num_workers=self.cfg.trainer.num_workers,
                                            train_filter_func=filter_relevant_ids,
                                            text_column=self.cfg.data.text_column,
                                            label_column=self.cfg.data.label_column)

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
    
    def get_all_tokens(self) -> Dict[int, List[str]]:
        token_dict: Dict[int, List[str]] = {}
        for entry in srsly.read_jsonl(self.data_dir / "train.jsonl"):
            token_dict[entry["id"]] = entry["tokens"]
        return token_dict
