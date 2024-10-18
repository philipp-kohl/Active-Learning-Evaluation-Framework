def import_registrable_components():
    import importlib
    importlib.import_module("ale.registry")
    importlib.import_module("ale.pipeline.pipeline_components")
    importlib.import_module("ale.teacher")
    importlib.import_module("ale.trainer")
    importlib.import_module("ale.corpus")
