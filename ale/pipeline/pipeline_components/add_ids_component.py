from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.pipeline_components.utils import create_path
from ale.preprocessing.add_id_to_jsonl import add_ids_to_jsonl


class AddIdsComponent(PipelineComponent):
    def prepare_run(self):
        data_cfg = self.pipeline_storage.cfg.data
        path = create_path(data_cfg.data_dir, data_cfg.train_file, data_cfg.file_format)

        self.store_function(
            add_ids_to_jsonl, file=path, output=path, start_id=0, force=False
        )
