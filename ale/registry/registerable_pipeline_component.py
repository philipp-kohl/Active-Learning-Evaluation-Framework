import logging

from ale.pipeline.pipeline_component import PipelineComponent
from ale.registry.registerable_component import ComponentRegistry

logger = logging.getLogger(__name__)


class PipelineComponentRegistry(ComponentRegistry):
    """
    This factory creates the correct profile component extractors by their name (ExtractorEnum)
    """

    VALUE_TYPE = PipelineComponent
