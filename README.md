# Active-Learning-Evaluation (ALE :beer:) Framework

The framework allows the comparison of different active learning strategies (queries) by simulating the annotation
process in the domain of natural language processing. This facilitates to build best practices for different use cases
and enables practitioners and researchers to make more informed decisions.

This is the repository for the paper [ALE: A Simulation-Based Active Learning Evaluation Framework for the Parameter-Driven Comparison of Query Strategies for NLP](https://doi.org/10.1007/978-3-031-39059-3_16).

Provides features:
- Sophisticated **configuration system** with [hydra](https://github.com/facebookresearch/hydra)
- **Experiment tracking** and documentation with [MLFlow](https://mlflow.org/)
- Easy to test **own strategies**: Write your own strategies by implementing the [base_teacher](ale/teacher/base_teacher.py)
- **Containerized** with [docker](https://www.docker.com/)
- **Resume experiments**: It's frustrating when long-running computations have to be completely restarted, thus we enable to
  resume paused or errored experiments
- **Reproducible**: Tracking parameters, models, git revision enables reproducible research
- Simulation with **different seeds**: to avoid "lucky punches" and inspect model's and strategy's stability
- Arbitrary usage of **datasets**
- Arbitrary usage of **ML/DL framework**: we provide a [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) implementation.
- Experiment with **cold start phase**: analog to the active learning strategy, the framework enables researcher to test
  strategies to select the first data increment.
- **Parallel computation**: different experiments can be run in parallel and report the results to a central MLFlow instance.
- **Custom pipeline steps**: ALE uses the pipeline architecture (see paper). The user can write own pipeline steps.

## Getting started

Prerequisite:
- [git](https://git-scm.com/)
- [poetry](https://python-poetry.org/docs/main/#installing-with-pipx)
- [docker](https://www.docker.com/)

Set up dependencies:

```
source ./dev-setup.sh ale-cuda.yaml
```

If poetry complains about not finding any specific version, try `poetry update`.

### Prepare machine for docker GPU usage

1. Install [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

2. Test
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Load Data

Run [load_conll](ale/data/huggingface_load_dataset_conll.py) and store the files in the data directory (data/conll2003).


## Run ALE locally:

```bash
mlflow ui
```

And in another terminal:

```bash
python ale/main.py
```

## Start Your Experiments with Docker

1. Start the tracking server:

```bash
docker compose -f docker-compose.mlflow.yaml up --build
```

2. Build your docker image:

```bash 
docker build -f DockerfileCUDA -t ale-cuda:1.0 .
```

3. Start your experiment:

Please replace `</absolute/path/to/your/data/folder>` with your data folder path in the following command:
```
docker run -it --network host \
--gpus '"device=0"' \
-v </absolute/path/to/your/data/folder>:/app/data/ \
ale-cuda:1.0 \
conda run --no-capture-output -n ale-cuda python ale/main.py teacher=randomizer mlflow.experiment_name=randomizer mlflow.url=http://localhost:5000
```

4. Shutdown mlflow services:
    1. Without deleting volumes
       ```bash
       docker compose -f docker-compose.mlflow.yaml down
       ```
    2. With deleting volumes
       ```bash
       docker compose -f docker-compose.mlflow.yaml down -v
       ```

### Configuration

The configuration will be instantiated into python objects (see ale/conf/).

| Config Group | Parameter Name        | Description                                                                                                                                                                                                                                                                            |
|--------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| converter    | converter_class       | Converter class registered via `PipelineComponentRegistry.register(''name'')`.                                                                                                                                                                                                         |
|              | target_format         | The converter component converts the data into this format.                                                                                                                                                                                                                            |
| data         | data_dir              | The directory the data resides in. The execution in the docker environment requires the base path data due to the mounting directory. In local execution the user can deviate from this convention.                                                                                    |
|              | train_file            | The training file's name without extension.                                                                                                                                                                                                                                            |
|              | dev_file              | The development file's name without extension.                                                                                                                                                                                                                                         |
|              | test_file             | The test file's name without extension.                                                                                                                                                                                                                                                |
|              | file_format           | The file format and the extension of the train, dev, and test file.                                                                                                                                                                                                                    |
|              | text_column           | The name of key in the json document which leads to the text information. Hint: Conventionally we use jsonl files. Default: text                                                                                                                                                       |
|              | label_column          | The name of key in the json document which leads to the label information. Corpora may include several label types such as document-level or sequence labels. See `trec_coarse` and `trec_fine in the configuration. Hint: Conventionally we use jsonl files. Default: label           |
|              | nlp_task              | The NLP task type. It will be validated against an enum. Allowed values at the moment are `CLS` and `NER`. The information helps for collecting labels, converting data and training.                                                                                                  |
| experiment   | initial_data_ratio    | The relative training data amount used for an initial model training.                                                                                                                                                                                                                  |
|              | initial_data_strategy | To inspect the cold start stage researchers can test different strategies analog to teachers with registered name.                                                                                                                                                                     |
|              | tracking_metrics      | List of metric names we want to use as measurement for strategy performance.                                                                                                                                                                                                           |
|              | step_size             | Denotes how many (absolute value) new data points a strategy should propose.                                                                                                                                                                                                           |
|              | seeds                 | List of integer seeds. The framework runs an isolated simulation (own corpus, trainer, strategy instances) for each seed.                                                                                                                                                              |
| mlflow       | url                   | The Mlflow tracking server's URL.                                                                                                                                                                                                                                                      |
|              | experiment_name       | Mlflow's top level concept represents experiments. We use these for each active learning strategy saparately.                                                                                                                                                                          |
|              | run_name              | Beneath each experiment, Mlflow tracks arbitrary runs. We can use the run names to describe our current approach under the experiment. This parameter is optional: If empty, the frameworks generates a run name based on the teacher strategy and a generated suffix with haikunator. |
| teacher      | strategy              | Teacher class registered via TeacherRegistry.register(''name'')`.                                                                                                                                                                                                                      |
|              | budget                | Represents the absolute number of training data points, which will be subselected for the teacher. The teacher may use the subset for prediction purposes. This avoids exploitation approaches to predict the whole unlabled dataset, which can be time-consuming.                     |
| technical    | use_gpu               | ID of the GPU. -1 defaults to CPU.                                                                                                                                                                                                                                                     |
|              | number_threads        | The number of concurrent threads used for parallel AL simulation for each seed. -1 for setting number of threads to number of seeds.                                                                                                                                                   |
|              | trainer_name          | Trainer class registered via `TrainerRegistry.register(''name'')`.                                                                                                                                                                                                                     |
|              | config_path           | Path to the config path relative to the project root. The config will be packaged into the docker image.                                                                                                                                                                               |
|              | corpus_manager        | Corpus class registered via CorpusRegistry.register(''name'')`.                                                                                                                                                                                                                        |
|              | language              | Language abbreviation (en, de, etc. )                                                                                                                                                                                                                                                  |
## Run tests with pytest
To run all tests in the repository, run `pytest .` in the root directory. 
If you want to run just one test, use `python -m pytest tests/<test_name>.py`

## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{kohlALESimulationBasedActive2023,
  title = {{{ALE}}: {{A Simulation-Based Active Learning Evaluation Framework}} for the {{Parameter-Driven Comparison}} of {{Query Strategies}} for {{NLP}}},
  shorttitle = {{{ALE}}},
  booktitle = {Deep {{Learning Theory}} and {{Applications}}},
  author = {Kohl, Philipp and Freyer, Nils and Kr{\"a}mer, Yoka and Werth, Henri and Wolf, Steffen and Kraft, Bodo and Meinecke, Matthias and Z{\"u}ndorf, Albert},
  editor = {Conte, Donatello and Fred, Ana and Gusikhin, Oleg and Sansone, Carlo},
  year = {2023},
  series = {Communications in {{Computer}} and {{Information Science}}},
  pages = {235--253},
  publisher = {{Springer Nature Switzerland}},
  address = {{Cham}},
  doi = {10.1007/978-3-031-39059-3\_16},
  copyright = {All rights reserved},
  isbn = {978-3-031-39059-3},
  langid = {english},
  keywords = {Active learning,Deep learning,Natural language processing,Query learning,Reproducible research}
}
```
