#!/bin/bash

default_gpu=0
default_run_name_suffix="v1"
dry_run=false

docker_image=philippkohl/active-learning-evaluation-framework
ale_version=2.4.2-dev
tracking_url="http://localhost:5000"

batch_size=64
label_smoothing=0.2
early_stopping_delta=0.0005
early_stopping_patience=5
max_epochs=30

variance_seeds="'[42, 4711, 815, 88, 26, 4, 928473, 1288, 204, 5187436957325, 44733425, 542990533, 314730904, 763884067, 235603638, 227058162, 786387716, 392716111, 800178982, 9963]'"

run_experiment() {
    local prefix=$1
    local log_file=$2
    local step_size=$3
    local dataset=$4
    local seeds=$5
    local stop_early=$6

    mlflow_experiment_name=$experiment_name
    mlflow_run_name="${prefix}_${run_name_suffix}"

    ale_command="docker run -it --network host --gpus \""device=$gpu\"" \
    -v /home/pk9678e/repositories/ner-corpora:/app/data/ $docker_image:$ale_version \
    conda run --no-capture-output -n ale-cuda python ale/main.py \
    data=$dataset teacher=$teacher mlflow.experiment_name=$mlflow_experiment_name mlflow.url=$tracking_url \
    mlflow.run_name=$mlflow_run_name trainer.batch_size=$batch_size \
    trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing \
    trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience \
    experiment.step_size=${step_size} teacher.sampling_budget=20000 experiment.assess_data_bias_eval_freq=5 \
    experiment.assess_overconfidence_eval_freq=5 experiment.seeds=$seeds"

    ale_command="$ale_command trainer.max_epochs=$max_epochs"

    if [ -n "$aggregation" ]; then
        ale_command="$ale_command teacher.aggregation_method=$aggregation"
    fi

    if [ "${stop_early}" = "true" ]; then
        ale_command="$ale_command experiment.stop_after_n_al_cycles=3"
    fi

    command="experiment run --send-mail --logging $log_file \"$ale_command\""

    if [ ${dry_run} = true ];
    then
      echo $command
    else
      eval $command
    fi
}

# Parse the command-line arguments
while getopts "t:a:g:e:r:d" option; do
   case "$option" in
       t) teacher=${OPTARG};;
       a) aggregation=${OPTARG};;
       g) gpu=${OPTARG};;
       e) experiment_name=${OPTARG};;
       r) run_name_suffix=${OPTARG};;
       d) dry_run=true;;
   esac
done

if [ -z "$gpu" ]; then gpu=$default_gpu; else echo "GPU: $gpu"; fi;
if [ -z "$run_name_suffix" ]; then run_name_suffix=$default_run_name_suffix; else echo "Run name suffix: $run_name_suffix"; fi;

if [ ${dry_run} = true ]; then
  echo "Start DRY run!"
fi

if [ -z "$experiment_name" ]; then
    echo "Error: Please provide experiment_name with -e" >&2
    exit 1
fi

echo -e "Parameters for the experiment:\n Ale version: $docker_image:$ale_version, \n Tracking URL: $tracking_url, \n \
    Batch size: $batch_size,\n Label smoothing: $label_smoothing, \n Early stopping delta: $early_stopping_delta, \n \
    Early stopping patience: $early_stopping_patience"

echo "Show test command with substitutions:"
mlflow_experiment_name=$experiment_name
mlflow_run_name="variance_test_conll_${run_name_suffix}"
echo "docker run -it --network host --gpus \""device=$gpu"\" -v /home/pk9678e/repositories/ner-corpora:/app/data/ $docker_image:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=conll2003 teacher=$teacher mlflow.experiment_name=$mlflow_experiment_name mlflow.url=$tracking_url mlflow.run_name=$mlflow_run_name trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=100 experiment.assess_overconfidence_eval_freq=100 experiment.stop_after_n_al_cycles=3 experiment.seeds='[42, 4711, 815, 88, 26, 4, 928473, 1288, 204, 5187436957325, 44733425, 542990533, 314730904, 763884067, 235603638, 227058162, 786387716, 392716111, 800178982, 9963]'"
echo "Press Enter to continue..."
read

echo "Start evaluation experiments for teacher $teacher with aggregation method $aggregation."

echo "Start variance test on CoNLL 2003 data."
run_experiment "variance_test_conll" "./variance_test_conll_$teacher.log" 500 "conll2003" "$variance_seeds" "true"
echo "Variance test on CoNLL 2003 data done."

echo "Start variance test on MedMentions data."
run_experiment "variance_test_medmentions" "./variance_test_medmentions_$teacher.log" 100 "medmentions" "$variance_seeds" "true"
echo "Variance test on MedMentions data done."
echo "Variance tests done."

echo "Start variance test on AURC data."
run_experiment "variance_test_aurc" "./variance_test_aurc_$teacher.log" 375 "aurc" "$variance_seeds" "true"
echo "Variance test on AURC data done."

echo "Start variance test on WNUT data."
run_experiment "variance_test_wnut" "./variance_test_wnut_$teacher.log" 500 "wnut16" "$variance_seeds" "true"
echo "Variance test on WNUT data done."

echo "Start variance test on SCIERC data."
run_experiment "variance_test_scierc" "./variance_test_scierc_$teacher.log" 75 "scierc" "$variance_seeds" "true"
echo "Variance test on SCIERC data done."

echo "Start variance test on JNLPBA data."
run_experiment "variance_test_jnlpba" "./variance_test_jnlpba_$teacher.log" 375 "jnlpba" "$variance_seeds" "true"
echo "Variance test on JNLPBA data done."

echo "Start variance test on GermEval data."
run_experiment "variance_test_germeval" "./variance_test_germeval_$teacher.log" 500 "germeval_14" "$variance_seeds" "true"
echo "Variance test on GermEval data done."

echo "Script done. Teacher $teacher evaluated. Have a look at results on the ML Flow tracking server :)"
