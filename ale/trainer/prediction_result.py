from typing import Dict, Optional, Union

from pydantic import BaseModel


class Span(BaseModel):
    start: int
    end: int
    label: str

    def __hash__(self):
        return hash(self.start) + hash(self.end) + hash(self.label)


class PredictionResult(BaseModel):
    classification_confidences: Optional[Dict[str, float]] = {}
    ner_confidences: Optional[Dict[Span, float]] = {}

    def add_ner(self, span: Span, score: float):
        if self.ner_confidences is None:
            self.ner_confidences = {}

        if span in self.ner_confidences:
            raise ValueError(f"Span ({span}) already in result set!")

        self.ner_confidences[span] = score

    def get_highest_confidence_label(self) -> Union[str, None]:
        if self.classification_confidences:
            return max(self.classification_confidences, key=self.classification_confidences.get)
        elif self.ner_confidences:
            return max(self.ner_confidences, key=self.ner_confidences.get).label
        else:
            return None
