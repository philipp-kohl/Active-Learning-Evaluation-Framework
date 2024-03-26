from typing import List, Dict

from ale.import_helper import import_registrable_components
from ale.config import NLPTask
from ale.teacher.exploitation.aggregation_methods import AggregationMethod
from ale.teacher.exploitation.margin_confidence import MarginTeacher
from ale.trainer.prediction_result import PredictionResult, TokenConfidence, LabelConfidence

import_registrable_components()
import pytest

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
        "Token 1": [0.3, 0.2, 0.5], # Margin: 0.2
        "Token 2": [0.1, 0.1, 0.5], # Margin: 0.4
        "Token 3": [0.4, 0.3, 0.8], # Margin: 0.4
    })

    predictions_2 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4], # Margin: 0.1
        "Token 2": [0.1, 0.1, 0.5], # Margin: 0.4
        "Token 3": [0.4, 0.3, 0.8], # Margin: 0.4
    })

    predictions_3 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4], # Margin: 0.1
        "Token 2": [0.1, 0.1, 0.5], # Margin: 0.4
        "Token 3": [0.4, 0.3, 0.8], # Margin: 0.4
    })

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }


def test_margin_for_ner_min(prediction_results: Dict[int, PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.MINIMUM)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results, 2)
    assert out_ids == [1, 2]


def test_margin_for_ner_avg(prediction_results: Dict[int, PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.AVERAGE)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results, 2)
    assert out_ids == [1, 2]


def test_margin_for_ner_max(prediction_results: Dict[int, PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.MAXIMUM)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results, 2)
    assert out_ids == [0, 1]


def test_margin_for_ner_std(prediction_results: Dict[int, PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.STD)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results, 2)
    assert out_ids == [0, 1]


def test_margin_for_ner_sum(prediction_results: Dict[int, PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.SUM)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results, 2)
    assert out_ids == [1, 2]
