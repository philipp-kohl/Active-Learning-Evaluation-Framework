from typing import List, Any, Dict
import random
import numpy as np
from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.predictor import Predictor
from ale.teacher.teacher_utils import bert_vectorize, get_cosine_similarity
from ale.trainer.prediction_result import TokenConfidence, PredictionResult


@TeacherRegistry.register("information-density")
class InformationDensityTeacher(BaseTeacher):
    """
    The information density teacher proposes samples that are most similar to unlabeled samples with entropy confidence in the predictions

    Applied to ER task:
        - Settles, B., Craven, M.: An analysis of active learning strategies for sequence labeling tasks. In: Proceedings of the 2008 
        Conference on Empirical Methods in Natural Language Processing. pp. 1070–1079 (2008)
        (no detailed information on feature vectors/embeddings ("vector representing the combination of the sequence attributes"), used with CRF)
        - Mendonca, V., Sardinha, A., Coheur, L., Santos, A.L.: Query Strategies, Assemble! Active Learning with Expert Advice for 
        Low-resource Natural Language Processing. In: 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE). pp. 1–8 (Jul 2020). 
        doi: 10.1109/FUZZ48607.2020.9177707
        (least confidence instead of entropy confidence)
        - Claveau, V., Kijak, E.: Strategies to select examples for active learning with conditional random fields. In: Lecture 
        Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics). 
        vol. 10761 LNCS, pp. 30–43 (2018). doi: 10.1007/978-3-319-77113-7 ̇3
        (no detailed information on feature vectors/embeddings, used with CRF)
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
        self.embeddings: List[np.ndarray] = bert_vectorize(corpus)
        self.corpus = corpus
        self.corpus_idx_list: List[int] = list(
            corpus.get_all_texts_with_ids().keys())

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        if budget < len(potential_ids):
            batch: List[int] = random.sample(potential_ids, budget)
        else:
            batch: List[int] = potential_ids
        
        # get entropy confidence for documents (inside budget)
        idx2text = self.corpus.get_text_by_ids(batch)
        prediction_results: Dict[int, PredictionResult] = self.predictor.predict(
            idx2text)
        uncertainty_scores: Dict[int, float] = self.compute_function(
            prediction_results, step_size)

        # get similarity score for each document (inside budget) in comparison to all unlabeled documents of the train corpus
        similarity_scores: Dict[int, float] = self.get_similarity_scores(
            batch, potential_ids)

        combined_scores: Dict[int, float] = {}
        for id in batch:
            combined_scores[id] = uncertainty_scores[id] * \
                similarity_scores[id]
        sorted_dict_by_combined_score = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True)
        out_ids: List[int] = [item[0]
                              for item in sorted_dict_by_combined_score[:step_size]]
        return out_ids

    def compute_ner(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, float]:
        """
        Entropy confidence is calculated on token-level and aggregated on instance-level as configured.
        """
        scores = dict()
        for idx, prediction in predictions.items():
            token_confidences: List[TokenConfidence] = prediction.ner_confidences_token
            confidence_scores: List[float] = []
            for token in token_confidences:
                confidences: List[float] = [
                    label_confidence.confidence for label_confidence in token.label_confidence]
                entropy = -sum([pk * np.log(pk) for pk in confidences])
                confidence_scores.append(entropy)
            instance_score = self.aggregate_function(confidence_scores)
            scores[idx] = instance_score
        return scores

    def compute_cls(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, int]:
        """ Not implemented
        """
        raise NotImplementedError(
            "Hybrid teacher is not implemented for text classification task.")

    def get_similarity_scores(self, batch: List[int], potential_ids: List[int]) -> Dict[int, float]:
        """ Compares bert embeddings of documents in batch with embeddings of all unannotated data points 
        by cosine similarity
        """
        scores: Dict[int, float] = {}
        unannotated_indices: List[int] = self.get_indices_for_embeddings(
            potential_ids)
        for id in batch:  # get score for all docs of the batch
            idx: int = self.get_index_for_embeddings(id)
            embedding: np.ndarray = self.embeddings[idx]
            # get cosine similarity to all unannotated data points
            similarity_scores: np.ndarray = [get_cosine_similarity(unannotated_embedding,embedding) for unannotated_embedding in self.embeddings[unannotated_indices]]
            # use average similarity score
            scores[id] = np.mean(similarity_scores)
        return scores

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
