from typing import List, Any
import random
from numpy.linalg import norm
import numpy as np
from ale.teacher.teacher_utils import embed_documents_with_tfidf
from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.predictor import Predictor


@TeacherRegistry.register("diversity")
class DiversityTeacher(BaseTeacher):
    """
    The diversity teacher proposes samples that are most dissimilar to already labeled samples (based on TF-IDF embeddings)

    Applied to ER task:
        - Chen, Y., Lasko, T.A., Mei, Q., Denny, J.C., Xu, H.: A study of active learning methods for named entity recognition
        in clinical text. Journal of Biomedical Informatics 58, 11–18 (Dec 2015). https://doi.org/10.1016/j.jbi.2015.09.010
        (in paper: embeddings based on word similarity)
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
        npm_tfidf: np.ndarray = self.embeddings.todense()
        for i in range(len(batch)):
            doc_id = batch[i]
            doc_vector: np.ndarray = npm_tfidf[self.get_index_for_embeddings(doc_id)]

            # calculate similarity score for doc with labeled corpus, use complete linkage: max cosine-similarity
            labeled_indices: List[int] = [
                self.get_index_for_embeddings(id) for id in annotated_ids]
            embeddings_annotated: List[np.ndarray] = npm_tfidf[labeled_indices]
            similarity_scores: np.ndarray = [np.dot(annotated_vector.reshape((annotated_vector.size,)),doc_vector.reshape((doc_vector.size),))/(norm(
                doc_vector)*norm(annotated_vector)) for annotated_vector in embeddings_annotated]

            # use max_sim as overall similarity score of the current doc to labeled dataset
            scores[doc_id] = max(similarity_scores)

        sorted_dict_by_score = sorted(
            scores.items(), key=lambda x: x[1])  # select items with minimal similarity to labeled docs

        out_ids = [item[0] for item in sorted_dict_by_score[:step_size]]
        return out_ids
