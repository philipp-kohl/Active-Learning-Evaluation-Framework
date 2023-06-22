from typing import Dict, Optional

from pydantic import BaseModel


class Span(BaseModel):
    start: int
    end: int
    label: str

    def __hash__(self):
        return hash(self.start) + hash(self.end) + hash(self.label)


class PredictionResult(BaseModel):
    classification_confidences: Optional[Dict[str, float]] = None
    ner_confidences: Optional[Dict[Span, float]] = None

    def add_ner(self, span: Span, score: float):
        if self.ner_confidences is None:
            self.ner_confidences = {}

        if span in self.ner_confidences:
            raise ValueError(f"Span ({span}) already in result set!")

        self.ner_confidences[span] = score
