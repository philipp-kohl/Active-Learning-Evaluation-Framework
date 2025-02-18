import pytest
from typing import List, Dict
from ale.config import NLPTask
from ale.import_helper import import_registrable_components
from ale.teacher.exploitation.aggregation_methods import AggregationMethod
from ale.teacher.exploitation.max_tag_count import MaxTagCountTeacher
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
        "Token 1": [0.3, 0.2, 0.5],  # 1
        "Token 2": [0.6, 0.1, 0.5],  # 0
        "Token 3": [0.4, 0.3, 0.8],  # 1
    })

    predictions_2 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4],  # 1
        "Token 2": [0.1, 0.1, 0.5],  # 1
        "Token 3": [0.4, 0.3, 0.8],  # 1
    })

    predictions_3 = create_prediction_result({
        "Token 1": [0.5, 0.2, 0.4],  # 0
        "Token 2": [0.1, 0.1, 0.5],  # 1
        "Token 3": [0.9, 0.3, 0.8],  # 0
    })

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }


def test_tag_count_for_ner_sum(prediction_results: Dict[int, PredictionResult]):
    entropy_teacher: MaxTagCountTeacher = MaxTagCountTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                             AggregationMethod.SUM)

    out_ids: List[int] = entropy_teacher.compute_function(
        prediction_results, 2)
    assert out_ids == [1, 0]
