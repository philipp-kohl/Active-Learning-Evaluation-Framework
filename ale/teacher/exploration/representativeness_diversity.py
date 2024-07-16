from typing import List, Any
import random
from numpy.linalg import norm
import numpy as np
from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.predictor import Predictor
from ale.teacher.teacher_utils import embed_documents_with_tfidf


@TeacherRegistry.register("representative-diversity")
class RepresentativeDiversityTeacher(BaseTeacher):
    """
    The representative diversity teacher proposes samples that are most dissimilar to already labeled samples and most similar to unlabeled samples (based on TF-IDF embeddings)

    Applied to ER task:
        - Kholghi, M., Sitbon, L., Zuccon, G., Nguyen, A.: External knowledge and query strategies in active learning: A study in 
        clinical information extraction. In: International Conference on Information and Knowledge Management, Proceedings. 
        vol. 19-23-Oct-2015, pp. 143–152 (2015). doi: 10.1145/2806416.280655
        (no information about embeddings used for equation (10))
    """

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task
        )
        self.k = len(self.labels)
        self.embeddings = embed_documents_with_tfidf(corpus=corpus)
        self.corpus_idx_list: List[int] = list(
            corpus.get_all_texts_with_ids().keys())
        self.corpus = corpus

    def get_index_for_embeddings(self, id: int) -> int:
        for i in range(len(self.corpus_idx_list)):
            if self.corpus_idx_list[i] == id:
                return i
        raise ValueError("Given id not in corpus.")

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        # only documents of the batch will be evaluated and sought for proposal
        if budget < len(potential_ids):
            batch: List[int] = random.sample(potential_ids, budget)
        else:
            batch: List[int] = potential_ids
        annotated_ids: List[int] = self.corpus.get_not_annotated_data_points_ids()
        scores = dict()

        # get the distance for each doc of the batch to all already labeled documents
        npm_tfidf = self.embeddings.todense()
        for i in range(len(batch)):
            doc_id = batch[i]
            doc_vector: np.ndarray = npm_tfidf[self.get_index_for_embeddings(doc_id)]

            # calculate diversity score for doc with labeled corpus, use average of all labeled docs: avg cosine-similarity
            labeled_indices: List[int] = [
                self.get_index_for_embeddings(id) for id in annotated_ids]
            embeddings_annotated: List[np.ndarray] = npm_tfidf[labeled_indices]
            diversity_scores: np.ndarray = [np.dot(annotated_vector.flatten(),doc_vector.flatten())/(norm(
                doc_vector)*norm(annotated_vector)) for annotated_vector in embeddings_annotated]

            # calculate representativeness score for doc with unlabeled docs, use average of all unlabeled docs: avg cosine-similarity
            unlabeled_indices: List[int] = [
                self.get_index_for_embeddings(id) for id in batch
            ]
            embeddings_not_annotated: List[np.ndarray] = npm_tfidf[unlabeled_indices]
            representative_scores: np.ndarray = [np.dot(not_annotated_vector.flatten(),doc_vector.flatten())/(norm(
                doc_vector)*norm(not_annotated_vector)) for not_annotated_vector in embeddings_not_annotated
            ]

            # use max_sim as overall similarity score of the current doc to labeled dataset
            scores[doc_id] = (1-np.mean(diversity_scores)) * \
                np.mean(representative_scores)

        sorted_dict_by_score = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)  # select items with minimal similarity to labeled docs (1-avg(diversity_scores)) and maximal similarity to unlabeled docs (avg(representative_scores))

        out_ids = [item[0] for item in sorted_dict_by_score[:step_size]]
        return out_ids
