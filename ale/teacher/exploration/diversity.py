from typing import List, Any, Dict
import random
import numpy as np
from ale.teacher.teacher_utils import embed_documents_with_tfidf, get_cosine_similarity
from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.predictor import Predictor
from sklearn.metrics.pairwise import cosine_similarity


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
        self.calculate_cosine_similarities()
        self.corpus = corpus

    def get_index_for_embeddings(self, id: int) -> int:
        for i in range(len(self.corpus_idx_list)):
            if self.corpus_idx_list[i] == id:
                return i
        raise ValueError("Given id not in corpus.")
    
    def get_indices_for_embeddings(self, ids: List[int]) -> List[int]:
        indices: List[int] = []
        for id in ids:
            for i in range(len(self.corpus_idx_list)):
                if self.corpus_idx_list[i] == id:
                    indices.append(i)
            raise ValueError("Given id"+str(id) + "not in corpus.")
        return indices

    def calculate_cosine_similarities(self) -> None:
        """ Calculates pairwise cosine similarity between all docs
        """
        self.cosine_similarities: np.ndarray = cosine_similarity(self.embeddings,self.embeddings)


    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        # only documents of the batch will be evaluated and sought for proposal
        if budget < len(potential_ids):
            batch: List[int] = random.sample(potential_ids, budget)
        else:
            batch: List[int] = potential_ids
        annotated_ids: List[int] = self.corpus.get_annotated_data_points_ids()
        scores = dict()

        # get the similarity for each doc of the batch to all already labeled documents
        for i in range(len(batch)):
            doc_id = batch[i]
            doc_idx = self.get_index_for_embeddings(doc_id)

            # get similarity score for doc with labeled corpus, use complete linkage: max cosine-similarity
            labeled_indices: List[int] = self.get_indices_for_embeddings(annotated_ids)
            similarity_scores: np.ndarray = self.cosine_similarities[doc_idx][labeled_indices]
            scores[doc_id] = similarity_scores.max()

        sorted_dict_by_score = sorted(
            scores.items(), key=lambda x: x[1])  # select items with minimal similarity to labeled docs

        out_ids = [item[0] for item in sorted_dict_by_score[:step_size]]
        return out_ids
