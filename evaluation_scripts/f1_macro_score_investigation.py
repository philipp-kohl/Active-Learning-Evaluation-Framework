import os

import mlflow
from mlflow.tracking import MlflowClient


def get_successful_subruns(experiment_id, run_id):
    client = MlflowClient()

    # Get the parent run
    parent_run = client.get_run(run_id)

    # List all runs under the experiment
    all_runs = client.search_runs(experiment_ids=[experiment_id])

    # Filter subruns which are children of the given run_id and successful
    successful_subruns = [
        run for run in all_runs
        if run.data.tags.get('mlflow.parentRunId') == run_id and run.info.status == 'FINISHED'
    ]

    return successful_subruns


def extract_test_metrics(run):
    test_metrics = {k: v for k, v in run.data.metrics.items() if k.startswith('test_f1')}
    return test_metrics


def calculate_mean_f1(metrics, without_o_tag=True):
    filter_list = ['test_f1_micro', 'test_f1_macro']
    if without_o_tag:
        filter_list.append('test_f1_O')
    # Remove micro and macro scores
    filtered_metrics = {k: v for k, v in metrics.items() if k not in filter_list}

    # Calculate mean of the remaining scores
    if filtered_metrics:
        mean_f1 = sum(filtered_metrics.values()) / len(filtered_metrics)
    else:
        mean_f1 = 0

    return mean_f1


def main():
    experiment_id = "338086452824637632"
    run_id = "b72c941bfa79455b90eacbfe3c60091c"

    # Get all successful subruns
    successful_subruns = get_successful_subruns(experiment_id, run_id)

    # Extract test metrics from each subrun
    all_test_metrics = {}
    for subrun in successful_subruns:
        subrun_id = subrun.info.run_id
        test_metrics = extract_test_metrics(subrun)
        all_test_metrics[subrun_id] = test_metrics

    for subrun_id, metrics in all_test_metrics.items():
        print(f"Subrun ID: {subrun_id}")
        mean_f1 = calculate_mean_f1(metrics)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
        print(f"Mean F1 Score: {mean_f1}")


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    main()
