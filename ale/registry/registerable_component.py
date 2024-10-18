import logging
from typing import Dict, Callable, TypeVar

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Common registry for all other registry types like trainer, teacher, corpus and pipeline
    """

    VALUE_TYPE = TypeVar("VALUE_TYPE")
    class_dictionary: Dict[str, VALUE_TYPE] = {}

    @classmethod
    def register(cls, type_name: str) -> Callable[[VALUE_TYPE], VALUE_TYPE]:
        """
        Registers a new trainer
        """

        def inner_wrapper(component: cls.VALUE_TYPE) -> cls.VALUE_TYPE:
            if (
                type_name in cls.class_dictionary
                and cls.class_dictionary[type_name].__name__ != component.__name__
            ):
                logger.warning(
                    f"'{type_name}' already registered for {cls.class_dictionary[type_name]}. "
                    f"Key will be overwritten with value '{component}'."
                )
            cls.class_dictionary[type_name] = component
            return component

        return inner_wrapper

    @classmethod
    def get_instance(cls, component_type: str) -> VALUE_TYPE:
        """
        Creates a new trainer of teacher_type
        """
        try:
            extractor = cls.class_dictionary[component_type]
        except KeyError as err:
            registered_types = cls.get_registered_types()
            raise ValueError(
                f"Unknown component. Known components are: {registered_types}"
            ) from err

        return extractor

    @classmethod
    def get_registered_types(cls):
        return [key for key, _ in cls.class_dictionary.items()]
