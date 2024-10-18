from ale.data.data import load_local_data
from ale.pipeline.pipeline_component import PipelineComponent


class LoadDataRawComponent(PipelineComponent):
    def prepare_run(self):
        data_cfg = self.pipeline_storage.cfg.data

        self.store_function(
            load_local_data,
            data_dir=data_cfg.data_dir,
            train_file=data_cfg.train_file,
            test_file=data_cfg.test_file,
            dev_file=data_cfg.dev_file,
            file_format=data_cfg.file_format,
        )
