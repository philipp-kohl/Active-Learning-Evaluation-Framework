#!/bin/bash

while getopts "t:a:" option; do
   case "$option" in
       t)teacher=${OPTARG};;
       a)aggregation=${OPTARG};;
   esac
done
if [ -z "$aggregation" ]; then aggregation="MINIMUM"; else echo "Aggregation method: $aggregation"; fi;

ale_version=1.6
tracking_url="http://localhost:5000"

batch_size=64
label_smoothing=0.2
early_stopping_delta=0.000005
early_stopping_patience=50

echo -e "Parameters for the experiment:\n Ale version: $ale_version, \n Tracking URL: $tracking_url, \n Batch size: $batch_size,\n Label smoothing: $label_smoothing, \n Early stopping delta: $early_stopping_delta, \n Early stopping patience: $early_stopping_patience"

echo "Start evaluation experiments for teacher $teacher with aggregation method $aggregation."

echo "Start variance test on CoNLL 2003 data."

experiment run --gpu-selection first --send-mail --logging ./variance_test_conll_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=conll2003 teacher=$teacher mlflow.experiment_name=variance_test_conll_$teacher mlflow.url=$tracking_url mlflow.run_name=variance_test_conll_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=100 experiment.assess_overconfidence_eval_freq=100 experiment.stop_after_n_al_cycles=3 experiment.seeds='[42, 4711, 815, 88, 26, 4, 928473, 1288, 204, 5187436957325, 44733425, 542990533, 314730904, 763884067, 235603638, 227058162, 786387716, 392716111, 800178982, 9963]'"

echo "Variance test on CoNLL 2003 data done."

echo "Start variance test on MedMentions data."

experiment run --gpu-selection first --send-mail --logging ./variance_test_medmentions_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=medmentions teacher=$teacher mlflow.experiment_name=variance_test_medmentions_$teacher mlflow.url=$tracking_url mlflow.run_name=variance_test_medmentions_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=100 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=100 experiment.assess_overconfidence_eval_freq=100 experiment.stop_after_n_al_cycles=3 experiment.seeds='[42, 4711, 815, 88, 26, 4, 928473, 1288, 204, 5187436957325, 44733425, 542990533, 314730904, 763884067, 235603638, 227058162, 786387716, 392716111, 800178982, 9963]'"

echo "Variance test on MedMentions data done."
echo "Variance tests done."

echo "Start performance tests."

echo "Start performance test on AURC data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_aurc_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=aurc teacher=$teacher mlflow.experiment_name=performance_test_aurc_$teacher mlflow.url=$tracking_url mlflow.run_name=performance_test_aurc_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=375 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on AURC data done."

echo "Start performance test on WNUT data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_wnut_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=wnut16 teacher=$teacher mlflow.experiment_name=performance_test_wnut_$teacher mlflow.url=$tracking_url mlflow.run_name=performance_test_wnut_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on WNUT data done."

echo "Start performance test on SCIERC data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_scierc_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=scierc teacher=$teacher mlflow.experiment_name=performance_test_scierc_$teacher mlflow.url=$tracking_url mlflow.run_name=performance_test_scierc_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=75 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on SCIERC data done."

echo "Start performance test on JNLPBA data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_jnlpba_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=jnlpba teacher=$teacher mlflow.experiment_name=performance_test_jnlpba_$teacher mlflow.url=$tracking_url mlflow.run_name=performance_test_jnlpba_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=375 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on JNLPBA data done."

echo "Start performance test on GermEval data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_germeval_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:$ale_version conda run --no-capture-output -n ale-cuda python ale/main.py data=germeval_14 teacher=$teacher mlflow.experiment_name=performance_test_germeval_$teacher mlflow.url=$tracking_url mlflow.run_name=performance_test_germeval_v1 trainer.batch_size=$batch_size trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=$label_smoothing trainer.early_stopping_delta=$early_stopping_delta trainer.early_stopping_patience=$early_stopping_patience experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on GermEval data done."

echo "Performance tests done."

echo "Script done. Teacher $teacher evaluated. Have a look at results on the ML Flow tracking server :)"
