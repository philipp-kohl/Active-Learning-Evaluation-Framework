from typing import List, Any, Dict
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.teacher.exploration.utils.embedding_helper import EmbeddingHelper
from ale.trainer.predictor import Predictor
from ale.teacher.teacher_utils import sentence_transformer_vectorize
from ale.trainer.prediction_result import TokenConfidence, PredictionResult
from ale.teacher.exploitation.aggregation_methods import AggregationMethod


@TeacherRegistry.register("information-density")
class InformationDensityTeacher(BaseTeacher):
    """
    The information density teacher proposes samples that are most similar to
    unlabeled samples with entropy confidence in the predictions

    Applied to ER task:
        - Settles, B., Craven, M.: An analysis of active learning strategies for sequence labeling tasks.
        In: Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing. pp. 1070–1079 (2008)
        (no detailed information on feature vectors/embeddings
        ("vector representing the combination of the sequence attributes"), used with CRF)
        - Mendonca, V., Sardinha, A., Coheur, L., Santos, A.L.:
        Query Strategies, Assemble! Active Learning with Expert Advice for Low-resource Natural Language Processing.
        In: 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE). pp. 1–8 (Jul 2020).
        doi: 10.1109/FUZZ48607.2020.9177707
        (least confidence instead of entropy confidence)
        - Claveau, V., Kijak, E.: Strategies to select examples for active learning with conditional random fields.
        In: Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and
        Lecture Notes in Bioinformatics). vol. 10761 LNCS, pp. 30–43 (2018). doi: 10.1007/978-3-319-77113-7 ̇3
        (no detailed information on feature vectors/embeddings, used with CRF)
    """

    def __init__(self, corpus: Corpus, predictor: Predictor, seed: int, labels: List[Any], nlp_task: NLPTask,
                 aggregation_method: AggregationMethod):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task,
            aggregation_method=aggregation_method
        )
        self.corpus_idx_list: List[int] = list(corpus.get_all_texts_with_ids().keys())
        self.embedding_helper = EmbeddingHelper(corpus, sentence_transformer_vectorize)
        self.cosine_similarities: np.ndarray = cosine_similarity(self.embedding_helper.get_embeddings())

    def propose(self, potential_ids: List[int], step_size: int,  budget: int) -> List[int]:
        if budget < len(potential_ids):
            batch: List[int] = random.sample(potential_ids, budget)
        else:
            batch: List[int] = potential_ids

        similarity_scores, uncertainty_scores = self.compute_partial_scores(batch, potential_ids)
        combined_scores = self.compute_combined_scores(batch, similarity_scores, uncertainty_scores)

        sorted_dict_by_combined_score = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        out_ids: List[int] = [item[0] for item in sorted_dict_by_combined_score[:step_size]]
        return out_ids

    def compute_combined_scores(self, batch, similarity_scores, uncertainty_scores) -> Dict[int, float]:
        combined_scores: Dict[int, float] = {
            idx: uncertainty_scores[idx] * similarity_scores[idx]
            for idx in batch
        }
        return combined_scores

    def compute_partial_scores(self, batch, potential_ids):
        # get entropy confidence for documents (inside budget)
        idx2text = self.corpus.get_text_by_ids(batch)
        prediction_results: Dict[int, PredictionResult] = self.predictor.predict(idx2text)
        uncertainty_scores: Dict[int, float] = self.compute_entropy(prediction_results)
        # get similarity score for each document (inside budget) in comparison
        # to all unlabeled documents of the train corpus
        similarity_scores: Dict[int, float] = self.get_similarity_scores(batch, potential_ids)
        return similarity_scores, uncertainty_scores

    def compute_entropy(self, predictions: Dict[int, PredictionResult]) -> Dict[int, float]:
        """
        Entropy confidence is calculated on token-level and aggregated on instance-level as configured.
        """
        scores: Dict[int, float] = {}
        for idx, prediction in predictions.items():
            confidence_scores: List[float] = [self.compute_token_entropy(token)
                                              for token in prediction.ner_confidences_token]
            instance_score = self.aggregate_function(confidence_scores)
            scores[idx] = instance_score
        return scores

    def compute_token_entropy(self, token):
        confidences: List[float] = [label_confidence.confidence for label_confidence in token.label_confidence]
        return -sum([pk * np.log(pk) for pk in confidences])

    def get_similarity_scores(self, batch: List[int], potential_ids: List[int]) -> Dict[int, float]:
        """ Compares bert embeddings of documents in batch with embeddings of all unannotated data points 
        by cosine similarity
        """
        scores: Dict[int, float] = {}
        unannotated_indices: List[int] = self.embedding_helper.get_embedding_indices_for_doc_ids(potential_ids)
        for doc_id in batch:  # get score for all docs of the batch
            embedding_index: int = self.embedding_helper.get_embedding_index_for_doc_id(doc_id)
            # get cosine similarity to all unannotated data points
            similarity_scores: np.ndarray = self.cosine_similarities[embedding_index][unannotated_indices]
            # use average similarity score
            scores[doc_id] = np.mean(similarity_scores)
        return scores

    def compute_ner(self, predictions: Dict[int, PredictionResult], step_size: int) -> List[int]:
        """
        Not implemented
        """
        raise NotImplementedError("Not used for this hybrid teacher")

    def compute_cls(self, predictions: Dict[int, PredictionResult], step_size: int) -> Dict[int, int]:
        """
        Not implemented
        """
        raise NotImplementedError("Hybrid teacher is not implemented for text classification task.")
