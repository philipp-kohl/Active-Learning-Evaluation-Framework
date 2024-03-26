import logging
from pathlib import Path
from typing import List, Union, Dict

import spacy
from spacy.tokens import Doc
from spacy.tokens import DocBin
from tqdm import tqdm

from ale.corpus.corpus import Corpus
from ale.registry.registerable_corpus import CorpusRegistry

logger = logging.getLogger(__name__)


@CorpusRegistry.register("spacy")
class SpacyIncrementalCorpus(Corpus):
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        try:
            Doc.set_extension("active_learning_id", default=-1)
        except ValueError as e:
            if not "[E090]" in str(e):
                raise e

        doc_bin = DocBin().from_disk(self.train_path)
        # model = spacy.blank(self.cfg.language) # TODO make multi language
        model = spacy.blank("en")
        logger.info("Start indexing spacy corpus")
        self.index: Dict[int, Doc] = {}
        docs = doc_bin.get_docs(model.vocab)
        for doc in tqdm(docs):
            self.index[doc._.active_learning_id] = doc
        logger.info("End indexing spacy corpus")

    def get_trainable_corpus(self):
        relevant_doc_bin = DocBin()

        for idx in self.relevant_ids:
            doc = self.index[idx]
            relevant_doc_bin.add(doc)

        return relevant_doc_bin

    def get_not_annotated_data_points_ids(self) -> List[int]:
        all_ids = set(self.index.keys())
        already_annotated = set(self.relevant_ids)

        return list(all_ids.difference(already_annotated))

    def get_all_texts_with_ids(self) -> Dict[int, str]:
        return {idx: doc.text for idx, doc in self.index.items()}

    def get_text_by_ids(self, idxs: List[int]) -> Dict[int, str]:
        return {idx: self.index[idx].text for idx in idxs}
