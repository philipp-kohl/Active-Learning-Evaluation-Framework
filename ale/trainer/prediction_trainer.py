from ale.trainer.base_trainer import BaseTrainer
from ale.trainer.predictor import Predictor


class PredictionTrainer(BaseTrainer, Predictor):
    """
    Trainer for prediction tasks.
    """
