import logging
from ale.registry.registerable_component import ComponentRegistry
# from ale.teacher.base_teacher import BaseTeacher

logger = logging.getLogger(__name__)


class TeacherRegistry(ComponentRegistry):
    """
    This factory creates the correct profile component extractors by their name (ExtractorEnum)
    """

    # VALUE_TYPE = BaseTeacher
