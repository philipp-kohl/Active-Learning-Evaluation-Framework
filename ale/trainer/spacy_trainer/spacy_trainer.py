import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Union, Optional, Any


import spacy
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

import ale.mlflowutils.mlflow_utils as mlflow_utils
from confection import Config
from mlflow import ActiveRun
from mlflow.entities import Run
from spacy import util, Language
from spacy.cli._util import setup_gpu, show_validation_error
from spacy.cli.evaluate import handle_scores_per_type
from spacy.tokens import Doc
from spacy.training.initialize import init_nlp
from spacy.training.loop import train as train_nlp
from wasabi import msg

from ale.utils import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_trainer import TrainerRegistry
from ale.trainer.base_trainer import MetricsType, PredictionTrainer
from ale.trainer.prediction_result import PredictionResult, Span

logger = logging.getLogger(__name__)

MODEL_PATH = "models/"
MODEL_DIR = "model-best/"


@TrainerRegistry.register("spacy-online-trainer")
class SpacyOnlineTrainer(PredictionTrainer):
    """ """

    SPANCAT_KEY: str = "sc"

    def __init__(
        self,
        dev_path: Path,
        test_path: Path,
        spacy_config: Path,
        use_gpu: int,
        seed: int,
        nlp_task: NLPTask,
        recreate_pipeline_each_run: bool
    ):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.use_gpu = use_gpu
        self.train_path = Path(self.temp_dir.name) / "train.spacy"
        self.dev_path = Path(dev_path)
        self.test_path = Path(test_path)
        self.config = spacy.util.load_config(spacy_config)
        self.nlp_task = nlp_task
        self.recreate_pipeline_each_run = recreate_pipeline_each_run

        predict_strategies = {
            NLPTask.CLS: self.get_classification_confidence_scores,
            NLPTask.NER: self.get_ner_confidence_scores,
        }

        self._prediction_strategy: Callable[
            [Doc], PredictionResult
        ] = predict_strategies[self.nlp_task]

        confs = {"paths": {"dev": str(dev_path), "train": str(self.train_path)}}
        self.config = self.config.merge(confs)
        self.config["system"]["seed"] = seed
        self.config["training"]["seed"] = seed

        self.spacy_config_path = Path(self.temp_dir.name) / "config.cfg"
        self.config.to_disk(self.spacy_config_path)

        logger.info(self.config)
        self.model_path = Path(self.temp_dir.name) / MODEL_PATH

        if not self.model_path.exists():
            self.model_path.mkdir(parents=True)
            msg.good(f"Created output directory: {self.model_path}")

        self.nlp: Optional[Language] = None
        self.dev_dataset = None
        self.test_dataset = None
        logger.info("Model initialized!")

    def train(self, train_corpus: Corpus, active_run: ActiveRun) -> Dict[str, any]:
        logger.info(f"Store training corpus at {self.train_path}")
        train_corpus.get_trainable_corpus().to_disk(self.train_path)
        # ale_spacy_logger = mlflow_logger(run_id=active_run.info.run_id)
        # spacy.registry.loggers.register(name="ale.mlflow_logger.v1", func=ale_spacy_logger)

        logger.info("Start training...")
        if self.nlp is None or self.recreate_pipeline_each_run:
            logger.info("First training. So let's initialize the nlp pipeline.")
            config = Config().from_disk(self.spacy_config_path)
            config["training"]["logger"]["run_id"] = active_run.info.run_id
            config.to_disk(self.spacy_config_path)
            logger.debug(config.to_str())
            self._train(
                config_path=self.spacy_config_path,
                output_path=self.model_path,
                use_gpu=self.use_gpu,
            )
        else:
            logger.info("Use cached nlp pipeline")
            new_config = Config().from_str(self.nlp.config.to_str())
            new_config = new_config.merge(self.config.interpolate())
            new_config["training"]["logger"]["run_id"] = active_run.info.run_id
            logger.debug(new_config.to_str())
            self.nlp.config.update(new_config)
            train_nlp(
                self.nlp,
                self.model_path,
                use_gpu=self.use_gpu,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        logger.info("End of training...")

        logger.info("Evaluate on dev set")
        metrics, dataset = self._evaluate(self.dev_path, cache=self.dev_dataset)
        logger.info(f"Evaluate on dev set finished")
        self.dev_dataset = dataset
        return metrics

    def evaluate(self) -> MetricsType:
        logger.info("Evaluate on test set")
        metrics, dataset = self._evaluate(self.test_path, cache=self.test_dataset)
        logger.info("Evaluate on test set finished")
        self.test_dataset = dataset

        return metrics

    def __del__(self):
        self.temp_dir.cleanup()

    def _train(
        self,
        config_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        *,
        use_gpu: int = -1,
        overrides: Dict[str, Any] = util.SimpleFrozenDict(),
    ):
        _config_path: Path = util.ensure_path(config_path)
        _output_path: Path = util.ensure_path(output_path)
        # Make sure all files and paths exists if they are needed
        if not _config_path or (str(_config_path) != "-" and not _config_path.exists()):
            msg.fail("Config file not found", str(config_path), exits=1)
        if not _output_path:
            msg.info("No output directory provided")
        else:
            if not _output_path.exists():
                _output_path.mkdir(parents=True)
                msg.good(f"Created output directory: {_output_path}")
            msg.info(f"Saving to output directory: {_output_path}")
        setup_gpu(use_gpu)
        with show_validation_error(_config_path):
            config = util.load_config(
                _config_path, overrides=overrides, interpolate=False
            )
        msg.divider("Initializing pipeline")
        with show_validation_error(_config_path, hint_fill=False):
            nlp = init_nlp(config, use_gpu=use_gpu)
        msg.good("Initialized pipeline")
        msg.divider("Training pipeline")
        self.nlp, _ = train_nlp(
            nlp, _output_path, use_gpu=use_gpu, stdout=sys.stdout, stderr=sys.stderr
        )

    def _evaluate(self, data_path, cache):
        spans_key: str = SpacyOnlineTrainer.SPANCAT_KEY

        if cache is None:
            logger.info(f"First time loading corpus: {data_path}")
            corpus = spacy.training.Corpus(data_path)
            dataset = list(corpus(self.nlp))
            logger.info(f"Corpus loaded")
        else:
            logger.info(f"Using cached corpus: {data_path}")
            dataset = cache

        scores = self.nlp.evaluate(dataset)

        metrics = {
            "TOK": "token_acc",
            "TAG": "tag_acc",
            "POS": "pos_acc",
            "MORPH": "morph_acc",
            "LEMMA": "lemma_acc",
            "UAS": "dep_uas",
            "LAS": "dep_las",
            "NER P": "ents_p",
            "NER R": "ents_r",
            "NER F": "ents_f",
            "TEXTCAT": "cats_score",
            "SENT P": "sents_p",
            "SENT R": "sents_r",
            "SENT F": "sents_f",
            "SPAN P": f"spans_{spans_key}_p",
            "SPAN R": f"spans_{spans_key}_r",
            "SPAN F": f"spans_{spans_key}_f",
            "SPEED": "speed",
        }
        results = {}
        data = {}
        for metric, key in metrics.items():
            if key in scores:
                if key == "cats_score":
                    metric = metric + " (" + scores.get("cats_score_desc", "unk") + ")"
                if isinstance(scores[key], (int, float)):
                    if key == "speed":
                        results[metric] = f"{scores[key]:.0f}"
                    else:
                        results[metric] = f"{scores[key] * 100:.2f}"
                else:
                    results[metric] = "-"
                data[re.sub(r"[\s/]", "_", key.lower())] = scores[key]

        data = handle_scores_per_type(scores, data, spans_key=spans_key, silent=True)

        return data, dataset

    def store_to_artifacts(self,  run: Run):
        model_best_path = str((self.model_path / "model-best").resolve())
        logger.info(f"Store model to: {model_best_path}")
        mlflow_utils.log_artifact(run, model_best_path, "best")

    def restore_from_artifacts(self, matching_run: Run):
        artifact_path = "best/model-best/"
        logger.info(f"Restore model from: {matching_run.info.run_id}/{artifact_path}")
        model_path = mlflow_utils.load_artifact(matching_run, artifact_path)
        self.nlp = spacy.load(model_path)

    def get_classification_confidence_scores(self, doc: Doc) -> PredictionResult:
        return PredictionResult(classification_confidences=doc.cats)

    def get_ner_confidence_scores(self, doc: Doc) -> PredictionResult:
        spans = doc.spans[SpacyOnlineTrainer.SPANCAT_KEY]
        scores = spans.attrs["scores"]

        result = PredictionResult()

        for span, score in zip(spans, scores):
            result.add_ner_span(
                Span(start=span.start_char, end=span.end_char, label=span.label_), score
            )

        return result

    def predict(self, docs: Dict[int, str]) -> Dict[int, PredictionResult]:
        results: Dict[int, PredictionResult] = dict()

        for idx, doc in zip(docs.keys(), self.nlp.pipe(docs.values())):
            prediction_result = self._prediction_strategy(doc)
            results[idx] = prediction_result

        return results

    def delete_artifacts(self, run: Run):
        repository = get_artifact_repository(run.info.artifact_uri)
        repository.delete_artifacts("best/model-best/")

