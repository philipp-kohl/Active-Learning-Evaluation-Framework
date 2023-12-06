from typing import List

from ale.config import AppConfig
from ale.corpus import SpacyIncrementalCorpus
from ale.corpus.corpus import Corpus
from ale.import_helper import import_registrable_components

import_registrable_components()

from pathlib import Path

import pytest
from hydra import initialize, compose

from ale.proposer.proposer_per_seed import AleBartenderPerSeed


@pytest.fixture
def config() -> AppConfig:
    with initialize(config_path="../test_config/"):
        cfg = compose(config_name="config", overrides=["experiment.step_size=10",
                                                       "teacher.sampling_budget=50",
                                                       "experiment.annotation_budget=205"])
        return cfg


@pytest.fixture
def create_seed_runner(config) -> AleBartenderPerSeed:
    return AleBartenderPerSeed(config,
                               4711,
                               Path("../artifacts/train.spacy"),
                               Path("../artifacts/dev.spacy"),
                               Path("../artifacts/test.spacy"),
                               ["Label-A", "Label-B"],
                               "4711",
                               "no",
                               config.experiment.tracking_metrics)


@pytest.fixture
def corpus() -> Corpus:
    return SpacyIncrementalCorpus(Path("../artifacts/train.spacy"))


def create_potential_ids(number: int):
    return range(number)


@pytest.mark.parametrize("name,current_corpus_size,potential_ids,expected_step_size,expected_sampling_budget",
                         [
                             ("Use normal step size", 100, create_potential_ids(500), 10, 50),
                             ("End of propsal", 100, create_potential_ids(5), 5, 5),
                             ("Exceed annotation budget", 200, create_potential_ids(500), 5, 50),
                             ("Remaining corpus < sampling budget", 100, create_potential_ids(48), 10, 48),
                         ])
def test_determine_step_size(create_seed_runner,
                             config,
                             name: str,
                             current_corpus_size: int,
                             potential_ids: List[int],
                             expected_step_size: int,
                             expected_sampling_budget: int):
    sampling_budget, step_size = create_seed_runner.determine_step_size(current_corpus_size, potential_ids)
    assert step_size == expected_step_size
    assert sampling_budget == expected_sampling_budget
