from typing import Dict, Any, Callable, TypeVar

import logging

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    This factory creates the correct profile component extractors by their name (ExtractorEnum)
    """

    VALUE_TYPE = TypeVar("VALUE_TYPE")
    class_dictionary: Dict[str, VALUE_TYPE] = {}

    @classmethod
    def register(cls, type_name: str) -> Callable[[VALUE_TYPE], VALUE_TYPE]:
        """
        Registers a new trainer
        """

        def inner_wrapper(trainer: cls.VALUE_TYPE) -> cls.VALUE_TYPE:
            if (
                type_name in cls.class_dictionary
                and cls.class_dictionary[type_name].__name__ != trainer.__name__
            ):
                logger.warning(
                    f"'{type_name}' already registered for {cls.class_dictionary[type_name]}. "
                    f"Key will be overwritten with value '{trainer}'."
                )
            cls.class_dictionary[type_name] = trainer
            return trainer

        return inner_wrapper

    @classmethod
    def get_instance(cls, teacher_type: str) -> VALUE_TYPE:
        """
        Creates a new trainer of teacher_type
        """
        try:
            extractor = cls.class_dictionary[teacher_type]
        except KeyError as err:
            registered_types = cls.get_registered_types()
            raise ValueError(
                f"Unknown component. Known components are: {registered_types}"
            ) from err

        return extractor

    @classmethod
    def get_registered_types(cls):
        return [key for key, _ in cls.class_dictionary.items()]
