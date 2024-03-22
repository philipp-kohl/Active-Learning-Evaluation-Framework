from typing import Dict, Optional, Union, List

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

    def __hash__(self):
        return hash((self.text, self.label_confidence))

    @property
    def predicted_label(self):
        return self.get_highest_confidence().label

    def get_highest_confidence(self) -> LabelConfidence:
        return max(self.label_confidence, key=lambda x: x.confidence)

    def get_lowest_confidence(self) -> LabelConfidence:
        return min(self.label_confidence, key=lambda x: x.confidence)


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
