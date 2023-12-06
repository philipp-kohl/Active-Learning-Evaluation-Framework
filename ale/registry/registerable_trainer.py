import logging
from ale.registry.registerable_component import ComponentRegistry
# from ale.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TrainerRegistry(ComponentRegistry):
    """
    This factory creates the correct profile component extractors by their name (ExtractorEnum)
    """

    # VALUE_TYPE = BaseTrainer
