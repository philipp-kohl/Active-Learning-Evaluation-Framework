import pytest
from typing import List, Dict
from ale.import_helper import import_registrable_components
from ale.config import NLPTask
from ale.teacher.exploitation.aggregation_methods import AggregationMethod
from ale.teacher.exploitation.tag_flip_historical import TagFlipTeacher, HistoricalSequence
from ale.trainer.prediction_result import PredictionResult, TokenConfidence, LabelConfidence

import_registrable_components()

LABELS = ["O", "B-PER", "B-ORG"]


def create_prediction_result(data: Dict[str, List[float]]) -> PredictionResult:
    token_confidences = []
    for token, confidence_per_label in data.items():
        label_confidence = [LabelConfidence(label=LABELS[idx], confidence=conf)
                            for idx, conf in enumerate(confidence_per_label)]
        token_confidence = TokenConfidence(text=token,
                                           label_confidence=label_confidence)
        token_confidences.append(token_confidence)

    return PredictionResult(ner_confidences_token=token_confidences)


@pytest.fixture
def prediction_results() -> Dict[int, PredictionResult]:
    predictions_1 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.5],  # ORG
        "Token 2": [0.1, 0.1, 0.5],  # ORG
        "Token 3": [0.4, 0.3, 0.8],  # ORG
    })

    predictions_2 = create_prediction_result({
        "Token 1": [0.3, 0.3, 0.4],  # ORG
        "Token 2": [0.1, 0.1, 0.5],  # ORG
        "Token 3": [0.4, 0.3, 0.8],  # ORG
    })

    predictions_3 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4],  # ORG
        "Token 2": [0.1, 0.1, 0.5],  # ORG
        "Token 3": [0.4, 0.3, 0.8],  # ORG
    })

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }


@pytest.fixture
def prediction_results_changed() -> Dict[int, PredictionResult]:
    predictions_1 = create_prediction_result({
        "Token 1": [0.3, 0.5, 0.2],  # PER
        "Token 2": [0.1, 0.1, 0.5],  # ORG
        "Token 3": [0.8, 0.3, 0.4],  # 0
    })  # Difference: 2

    predictions_2 = create_prediction_result({
        "Token 1": [0.3, 0.3, 0.4],  # ORG
        "Token 2": [0.1, 0.1, 0.5],  # ORG
        "Token 3": [0.8, 0.3, 0.4],  # 0
    })  # Difference: 1

    predictions_3 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4],  # ORG
        "Token 2": [0.1, 0.1, 0.5],  # ORG
        "Token 3": [0.4, 0.3, 0.8],  # ORG
    })  # Difference: 0

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }


def test_initial_train(prediction_results: Dict[int, PredictionResult]):
    tag_flip_teacher: TagFlipTeacher = TagFlipTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                      AggregationMethod.SUM)
    assert tag_flip_teacher.historical_sequence == None
    out_ids: List[int] = tag_flip_teacher.compute_function(
        prediction_results, 2)
    assert tag_flip_teacher.historical_sequence.__class__ == HistoricalSequence


def test_tag_flip_for_sum(prediction_results: Dict[int, PredictionResult], prediction_results_changed: Dict[int, PredictionResult]):
    tag_flip_teacher: TagFlipTeacher = TagFlipTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                      AggregationMethod.SUM)
    assert tag_flip_teacher.historical_sequence == None
    out_ids: List[int] = tag_flip_teacher.compute_function(
        prediction_results, 2)
    assert tag_flip_teacher.historical_sequence.__class__ == HistoricalSequence
    out_ids: List[int] = tag_flip_teacher.compute_function(
        prediction_results_changed, 2)
    assert out_ids == [0, 1]
