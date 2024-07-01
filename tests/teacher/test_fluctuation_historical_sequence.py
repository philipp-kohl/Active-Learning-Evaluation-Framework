from typing import List, Dict

from ale.import_helper import import_registrable_components
from ale.config import NLPTask
from ale.teacher.exploitation.aggregation_methods import AggregationMethod
from ale.teacher.exploitation.fluctuation_historical_sequence import FluctuationHistoricalSequenceTeacher, HistoricalSequence
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
        "Token 1": [0.3, 0.2, 0.5], # LC: 0.5
        "Token 2": [0.1, 0.1, 0.5], # LC: 0.5
        "Token 3": [0.4, 0.3, 0.8], # LC: 0.2
    }) # Max: 0.5

    predictions_2 = create_prediction_result({
        "Token 1": [0.3, 0.3, 0.4], # LC: 0.6
        "Token 2": [0.1, 0.1, 0.5], # LC: 0.5
        "Token 3": [0.4, 0.3, 0.8], # LC: 0.2
    }) # Max: 0.6

    predictions_3 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4], # LC: 0.6
        "Token 2": [0.1, 0.1, 0.5], # LC: 0.5
        "Token 3": [0.4, 0.3, 0.8], # LC: 0.2
    }) # Max: 0.6

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }

@pytest.fixture
def prediction_results_changed() -> Dict[int, PredictionResult]:
    predictions_1 = create_prediction_result({
        "Token 1": [0.3, 0.5, 0.2],  # LC: 0.5
        "Token 2": [0.1, 0.1, 0.5],  # LC: 0.5
        "Token 3": [0.3, 0.3, 0.3],  # LC: 0.7
    }) # Max: 0.7

    predictions_2 = create_prediction_result({
        "Token 1": [0.3, 0.3, 0.5], # LC: 0.5
        "Token 2": [0.1, 0.1, 0.5], # LC: 0.5
        "Token 3": [0.8, 0.3, 0.4], # LC: 0.2
    }) # Max. 0.5

    predictions_3 = create_prediction_result({
        "Token 1": [0.3, 0.2, 0.4], # LC: 0.6
        "Token 2": [0.1, 0.1, 0.5], # LC: 0.5
        "Token 3": [0.4, 0.3, 0.8], # LC: 0.2
    }) # Max. 0.6

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }


def test_initial_train(prediction_results: Dict[int, PredictionResult]):
    fluct_teacher: FluctuationHistoricalSequenceTeacher = FluctuationHistoricalSequenceTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.MAXIMUM) # take max now (similar to min with LC Teacher)
    assert fluct_teacher.historical_sequence == None
    out_ids: List[int] = fluct_teacher.compute_function(prediction_results, 2)
    assert fluct_teacher.historical_sequence.__class__ == HistoricalSequence

def test_tag_flip_for_sum(prediction_results: Dict[int, PredictionResult], prediction_results_changed: Dict[int,PredictionResult]):
    fluct_teacher: FluctuationHistoricalSequenceTeacher = FluctuationHistoricalSequenceTeacher(None, None, 0, LABELS, NLPTask.NER,
                                                  AggregationMethod.MAXIMUM)
    assert fluct_teacher.historical_sequence == None
    out_ids: List[int] = fluct_teacher.compute_function(prediction_results, 2)
    assert fluct_teacher.historical_sequence.__class__ == HistoricalSequence
    out_ids: List[int] = fluct_teacher.compute_function(prediction_results_changed,2)
    assert out_ids == [0,2]
