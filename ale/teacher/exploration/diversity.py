import random
from typing import List, Any, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.teacher.exploration.utils.embedding_helper import EmbeddingHelper
from ale.teacher.teacher_utils import tfidf_vectorize
from ale.trainer.predictor import Predictor


@TeacherRegistry.register("diversity")
class DiversityTeacher(BaseTeacher):
    """
    The diversity teacher proposes samples that are most dissimilar to already labeled samples
    (based on TF-IDF embeddings)

    Applied to ER task:
        - Chen, Y., Lasko, T.A., Mei, Q., Denny, J.C., Xu, H.: A study of active learning methods for named entity
        recognition in clinical text. Journal of Biomedical Informatics 58, 11–18 (Dec 2015).
        https://doi.org/10.1016/j.jbi.2015.09.010
        In paper: embeddings based on different similarities (word, syntax, semantic, combined)
        This implementation: solely on word similarity (TF-IDF)
    """

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task
        )
        self.embedding_helper = EmbeddingHelper(corpus, tfidf_vectorize)
        self.cosine_similarities: np.ndarray = cosine_similarity(self.embedding_helper.get_embeddings())

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        # only documents of the batch will be evaluated and sought for proposal
        if budget < len(potential_ids):
            batch: List[int] = random.sample(potential_ids, budget)
        else:
            batch: List[int] = potential_ids

        annotated_ids: List[int] = self.corpus.get_annotated_data_points_ids()
        if len(annotated_ids) == 0:
            raise NotImplementedError("This active learning strategy can not be used for initial data proposal.")

        scores: Dict[int, float] = self.compute_similarity_scores_for_docs(annotated_ids, batch)

        # select items with minimal similarity to labeled docs
        sorted_dict_by_score = sorted(scores.items(), key=lambda x: x[1])
        out_ids = [item[0] for item in sorted_dict_by_score[:step_size]]
        return out_ids

    def compute_similarity_scores_for_docs(self, annotated_ids: List[int], batch: List[int]):
        scores: Dict[int, float] = {}
        labeled_data_embedding_indices: List[int] = (self.embedding_helper.
                                                     get_embedding_indices_for_doc_ids(annotated_ids))
        # get the similarity for each doc of the batch to all already labeled documents
        for doc_id in batch:
            scores[doc_id] = self.compute_similarity_score_for_single_doc(doc_id, labeled_data_embedding_indices)

        return scores

    def compute_similarity_score_for_single_doc(self, doc_id: int, labeled_indices: List[int]) -> float:
        embedding_idx = self.embedding_helper.get_embedding_index_for_doc_id(doc_id)
        similarity_scores: np.ndarray = self.cosine_similarities[embedding_idx][labeled_indices]
        # get similarity score for doc with labeled corpus, use complete linkage: max cosine-similarity
        return similarity_scores.max()




