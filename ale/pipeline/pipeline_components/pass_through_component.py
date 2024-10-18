import logging
import os
import shutil
import tempfile

import mlflow

from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.pipeline_components.utils import prepare_data
from ale.registry.registerable_pipeline_component import PipelineComponentRegistry

logger = logging.getLogger(__name__)


@PipelineComponentRegistry.register("pass_through")
class PassThroughComponent(PipelineComponent):
    def pass_through(self, train_path, dev_path, test_path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_train_path = os.path.join(tmp_dir, os.path.basename(train_path))
            dest_dev_path = os.path.join(tmp_dir, os.path.basename(dev_path))
            dest_test_path = os.path.join(tmp_dir, os.path.basename(test_path))

            # Copy the files to the temporary directory
            shutil.copy(train_path, dest_train_path)
            shutil.copy(dev_path, dest_dev_path)
            shutil.copy(test_path, dest_test_path)

            mlflow.log_artifacts(str(tmp_dir), "data")

    def prepare_run(self):
        train_path, dev_path, test_path = prepare_data(self.pipeline_storage.cfg)

        self.store_function(self.pass_through,
                            train_path=train_path,
                            dev_path=dev_path,
                            test_path=test_path)
