from enum import Enum

class NLPTask(str, Enum):
    CLS = "CLS"
    NER = "NER"