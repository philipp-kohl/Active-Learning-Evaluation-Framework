from typing import Dict, Optional, Union, List
from ale.teacher.teacher_utils import is_named_entity
from pydantic import BaseModel


class Span(BaseModel):
    start: int
    end: int
    label: str

    def __hash__(self):
        return hash(self.start) + hash(self.end) + hash(self.label)


class LabelConfidence(BaseModel):
    label: str
    confidence: float

    def __hash__(self):
        return hash((self.label, self.confidence))


class TokenConfidence(BaseModel):
    text: str
    label_confidence: List[LabelConfidence]
    gold_label: Optional[str] = None
    predicted_label: Optional[str] = None

    def __hash__(self):
        return hash((self.text, self.label_confidence))

    def get_confidence_for_label(self, label: str) -> float:
        for label_confidence in self.label_confidence:
            if label_confidence.label == label:
                return label_confidence.confidence

    def get_predicted_label(self) -> str:
        """
        If no predicted label was set, the label with the highest confidence score is returned.
        The label will be set in case of a CRF classifier because the highest confidence label might not be the predicted label.
        """
        if not self.predicted_label:
            self.predicted_label = max(self.label_confidence, key=lambda x: x.confidence).label
        return self.predicted_label

    def get_confidence_for_predicted_label(self) -> float:
        return self.get_confidence_for_label(self.predicted_label)

    def get_highest_confidence(self) -> LabelConfidence:
        return max(self.label_confidence, key=lambda x: x.confidence)

    def get_lowest_confidence(self) -> LabelConfidence:
        return min(self.label_confidence, key=lambda x: x.confidence)

    def get_highest_k(self, k: int) -> List[LabelConfidence]:
        if len(self.label_confidence) < k:
            raise Exception(f"Get top k ({k}) labels by confidence exceeds number of labels!")

        return sorted(self.label_confidence, key=lambda x: x.confidence, reverse=True)[:k]


class PredictionResult(BaseModel):
    classification_confidences: Optional[Dict[str, float]] = {}
    ner_confidences_span: Optional[Dict[Span, float]] = {}
    ner_confidences_token: Optional[List[TokenConfidence]] = []

    def add_ner_span(self, span: Span, score: float):
        if self.ner_confidences_span is None:
            self.ner_confidences_span = {}

        if span in self.ner_confidences_span:
            raise ValueError(f"Span ({span}) already in result set!")

        self.ner_confidences_span[span] = score

    def get_highest_confidence_label(self) -> Union[str, None]:
        if self.classification_confidences:
            return max(self.classification_confidences, key=self.classification_confidences.get)
        elif self.ner_confidences_span:
            return max(self.ner_confidences_span, key=self.ner_confidences_span.get).label
        else:
            return None

    def get_all_label_classes(self) -> List[str]:
        if self.classification_confidences:
            return list(self.classification_confidences.keys())
        elif self.ner_confidences_span:
            return list(self.ner_confidences_span.keys())
        else:
            token_confidence: TokenConfidence = self.ner_confidences_token[0]
            label_confidences: List[LabelConfidence] = token_confidence.label_confidence

            return [conf.label for conf in label_confidences if is_named_entity(conf.label)]
