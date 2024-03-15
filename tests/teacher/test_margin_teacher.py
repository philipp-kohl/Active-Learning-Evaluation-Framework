from typing import List, Dict

from ale.config import NLPTask
from ale.import_helper import import_registrable_components
from ale.teacher.exploitation.aggregation_methods import AGGREGATION_METHOD
from ale.teacher.exploitation.margin_confidence import MarginTeacher
from ale.trainer.prediction_result import PredictionResult

import_registrable_components()
import pytest


@pytest.fixture
def prediction_results() -> Dict[int,PredictionResult]:
    ner_confidences_token_1 = {
        0: [0.3, 0.2, 0.5],
        1: [0.1, 0.1, 0.5],
        2: [0.4, 0.3, 0.8],
    }
    predictions_1: PredictionResult = PredictionResult()
    predictions_1.ner_confidences_token = ner_confidences_token_1

    ner_confidences_token_2 = {
        0: [0.3, 0.2, 0.4],
        1: [0.1, 0.1, 0.5],
        2: [0.4, 0.3, 0.8],
    }
    predictions_2: PredictionResult = PredictionResult()
    predictions_2.ner_confidences_token = ner_confidences_token_2

    ner_confidences_token_3 = {
        0: [0.3, 0.2, 0.4],
        1: [0.1, 0.1, 0.5],
        2: [0.4, 0.3, 0.8],
    }
    predictions_3: PredictionResult = PredictionResult()
    predictions_3.ner_confidences_token = ner_confidences_token_3

    return {
        0: predictions_1,
        1: predictions_2,
        2: predictions_3,
    }

def test_margin_for_ner_min(prediction_results: List[PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, ["org", "per", "loc"], NLPTask.NER, AGGREGATION_METHOD.MINIMUM)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results,2)
    assert out_ids==[1,2]

def test_margin_for_ner_avg(prediction_results: List[PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, ["org", "per", "loc"], NLPTask.NER, AGGREGATION_METHOD.AVERAGE)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results,2)
    assert out_ids==[1,2]

def test_margin_for_ner_max(prediction_results: List[PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, ["org", "per", "loc"], NLPTask.NER, AGGREGATION_METHOD.MAXIMUM)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results,2)
    assert out_ids==[0,1]

def test_margin_for_ner_std(prediction_results: List[PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, ["org", "per", "loc"], NLPTask.NER, AGGREGATION_METHOD.STD)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results,2)
    assert out_ids==[0,1]

def test_margin_for_ner_sum(prediction_results: List[PredictionResult]):
    margin_teacher: MarginTeacher = MarginTeacher(None, None, 0, ["org", "per", "loc"], NLPTask.NER, AGGREGATION_METHOD.SUM)

    out_ids: List[int] = margin_teacher.compute_function(prediction_results,2)
    assert out_ids==[1,2]