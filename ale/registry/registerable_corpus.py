import logging

from ale.corpus.corpus import Corpus
from ale.registry.registerable_component import ComponentRegistry

logger = logging.getLogger(__name__)


class CorpusRegistry(ComponentRegistry):
    """
    This factory creates the correct profile component extractors by their name (ExtractorEnum)
    """

    VALUE_TYPE = Corpus
