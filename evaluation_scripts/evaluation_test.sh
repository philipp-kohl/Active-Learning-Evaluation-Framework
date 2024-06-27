#!/bin/bash

while getopts "t:a:" option; do
   case "$option" in
       t)teacher=${OPTARG};;
       a)aggregation=${OPTARG};;
   esac
done

echo "$aggregation"

echo "Start evaluation experiments for teacher $teacher."

echo "Start variance test on CoNLL 2003 data."

experiment run --gpu-selection first --send-mail --logging ./variance_test_conll_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=conll2003 teacher=$teacher mlflow.experiment_name=variance_test_conll_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=variance_test_conll_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=100 experiment.assess_overconfidence_eval_freq=100 experiment.stop_after_n_al_cycles=3 experiment.seeds='[42, 4711, 815, 88, 26, 4, 928473, 1288, 204, 5187436957325, 44733425, 542990533, 314730904, 763884067, 235603638, 227058162, 786387716, 392716111, 800178982, 9963]'"

echo "Variance test on CoNLL 2003 data done."

echo "Start variance test on MedMentions data."

experiment run --gpu-selection first --send-mail --logging ./variance_test_medmentions_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=medmentions teacher=$teacher mlflow.experiment_name=variance_test_medmentions_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=variance_test_medmentions_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=100 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=100 experiment.assess_overconfidence_eval_freq=100 experiment.stop_after_n_al_cycles=3 experiment.seeds='[42, 4711, 815, 88, 26, 4, 928473, 1288, 204, 5187436957325, 44733425, 542990533, 314730904, 763884067, 235603638, 227058162, 786387716, 392716111, 800178982, 9963]'"

echo "Variance test on MedMentions data done."
echo "Variance tests done."

echo "Start performance tests."

echo "Start performance test on AURC data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_aurc_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=aurc teacher=$teacher mlflow.experiment_name=performance_test_aurc_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=performance_test_aurc_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=375 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on AURC data done."

echo "Start performance test on WNUT data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_wnut_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=wnut16 teacher=$teacher mlflow.experiment_name=performance_test_wnut_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=performance_test_wnut_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on WNUT data done."

echo "Start performance test on SCIERC data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_scierc_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=scierc teacher=$teacher mlflow.experiment_name=performance_test_scierc_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=performance_test_scierc_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=75 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on SCIERC data done."

echo "Start performance test on JNLPBA data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_jnlpba_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=jnlpba teacher=$teacher mlflow.experiment_name=performance_test_jnlpba_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=performance_test_jnlpba_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=375 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on JNLPBA data done."

echo "Start performance test on GermEval data."
experiment run --gpu-selection first --send-mail --logging ./performance_test_germeval_$teacher.log "docker run -it --network host --gpus '"device=0"' -v /home/pk9678e/repositories/ner-corpora:/app/data/ ale-cuda:1.6 conda run --no-capture-output -n ale-cuda python ale/main.py data=germeval_14 teacher=$teacher mlflow.experiment_name=performance_test_germeval_$teacher mlflow.url=http://localhost:5000 mlflow.run_name=performance_test_germeval_v1 trainer.batch_size=64 trainer.huggingface_model=distilbert/distilroberta-base trainer.label_smoothing=0.2 trainer.early_stopping_delta=0.000005 trainer.early_stopping_patience=50 experiment.step_size=500 teacher.sampling_budget=20000 teacher.aggregation_method=$aggregation experiment.assess_data_bias_eval_freq=5 experiment.assess_overconfidence_eval_freq=5 experiment.seeds='[42, 4711, 815]"
echo "Performance test on GermEval data done."

echo "Performance tests done."

echo "Script done. Teacher $teacher evaluated. Have a look at results on the ML Flow tracking server :)"



