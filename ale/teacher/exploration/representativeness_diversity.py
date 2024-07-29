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


@TeacherRegistry.register("representative-diversity")
class RepresentativeDiversityTeacher(BaseTeacher):
    """
    The representative diversity teacher proposes samples that are most dissimilar to already labeled samples and most
    similar to unlabeled samples (based on TF-IDF embeddings)

    Applied to ER task:
        - Kholghi, M., Sitbon, L., Zuccon, G., Nguyen, A.: External knowledge and query strategies in active learning:
        A study in clinical information extraction.
        In: International Conference on Information and Knowledge Management, Proceedings. vol. 19-23-Oct-2015,
        pp. 143–152 (2015). doi: 10.1145/2806416.280655

        Notes:
          - "We employed a typical feature set, including linguistic (part-of-speech tags), orthographical
            (regular expression patterns), lexical and morphological (suffixes/prefixes and character n-grams),
            contextual (window of k words), and semantic features. The Medtex system [18] was leveraged to extract
            semantic features" cited from paper
          - Not enough information to reproduce the feature set. Partially very med specific. Fallback to tfidf.
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

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:
        if budget < len(potential_ids):
            batch: List[int] = random.sample(potential_ids, budget)
        else:
            batch: List[int] = potential_ids

        annotated_ids: List[int] = self.corpus.get_annotated_data_points_ids()

        if len(annotated_ids) == 0:
            raise NotImplementedError("This active learning strategy can not be used for initial data proposal.")

        scores: Dict[int, float] = self.compute_scores(annotated_ids, batch)

        # select items with minimal similarity to labeled docs (1-avg(diversity_scores))
        # and maximal similarity to unlabeled docs (avg(representative_scores))
        sorted_dict_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out_ids = [item[0] for item in sorted_dict_by_score[:step_size]]
        return out_ids

    def compute_scores(self, annotated_ids: List[int], batch: List[int]) -> Dict[int, float]:
        labeled_embedding_indices: List[int] = self.embedding_helper.get_embedding_indices_for_doc_ids(annotated_ids)
        unlabeled_embedding_indices: List[int] = self.embedding_helper.get_embedding_indices_for_doc_ids(
            self.corpus.get_not_annotated_data_points_ids())
        scores: Dict[int, float] = {}
        # get the similarity for each doc of the batch to all already labeled documents
        for doc_id in batch:
            scores[doc_id] = self.compute_single_score(doc_id, labeled_embedding_indices, unlabeled_embedding_indices)
        return scores

    def compute_single_score(self,
                             doc_id: int,
                             labeled_embedding_indices: List[int],
                             unlabeled_embedding_indices: List[int]) -> float:
        doc_idx = self.embedding_helper.get_embedding_index_for_doc_id(doc_id)
        # calculate representativeness score for doc with unlabeled docs,
        # use average of all unlabeled docs: avg cosine-similarity (see formula 5)
        representative_scores: np.ndarray = self.cosine_similarities[doc_idx][unlabeled_embedding_indices]
        # get similarity score for doc with labeled corpus,
        # use average of all labeled docs: avg cosine-similarity (see formula 8)
        diversity_scores: np.ndarray = self.cosine_similarities[doc_idx][labeled_embedding_indices]
        # use max_sim as overall similarity score of the current doc to labeled dataset (see formula 10)
        return np.mean(representative_scores) * (1 - np.mean(diversity_scores))
