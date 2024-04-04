import torch
import torchmetrics


def derive_labels(labels):
    def enumerate_v2(xs, start=0, step=1):
        for x in xs:
            yield start, x
            start += step

    general_label_id_mapping = {0: 0}
    id2label = {0: "O"}
    for idx, label in enumerate_v2(labels, start=1, step=2):
        id2label[idx] = f"B-{label}"
        id2label[idx + 1] = f"I-{label}"

    for idx, label in id2label.items():
        if label is not 'O':
            label_wo_bio = label.lstrip("B-").lstrip("I-")
            general_label_id_mapping[idx] = labels.index(label_wo_bio) + 1

    label2id = {value: key for key, value in id2label.items()}
    return id2label, label2id, general_label_id_mapping


def is_valid_for_prog_bar(metric_name: str):
    return "f1_macro" in metric_name.lower() or "f1_micro" in metric_name.lower()


def create_metrics(num_labels: int):
    return {
        "precision_micro": torchmetrics.Precision(task="multiclass",
                                                  num_classes=num_labels,
                                                  average='micro',
                                                  ignore_index=-1),
        "recall_micro": torchmetrics.Recall(task="multiclass",
                                            num_classes=num_labels,
                                            average='micro',
                                            ignore_index=-1),
        "f1_micro": torchmetrics.F1Score(task="multiclass",
                                         num_classes=num_labels,
                                         average='micro',
                                         ignore_index=-1),
        "precision_macro": torchmetrics.Precision(task="multiclass",
                                                  num_classes=num_labels,
                                                  average='macro',
                                                  ignore_index=-1),
        "recall_macro": torchmetrics.Recall(task="multiclass",
                                            num_classes=num_labels,
                                            average='macro',
                                            ignore_index=-1),
        "f1_macro": torchmetrics.F1Score(task="multiclass",
                                         num_classes=num_labels,
                                         average='macro',
                                         ignore_index=-1)
    }


class LabelGeneralizer:
    def __init__(self, bio_id_to_coarse_label_id, device: str = "cpu"):
        # Convert the mapping to a PyTorch tensor for efficient indexing
        max_id = max(bio_id_to_coarse_label_id.keys())
        self.mapping_tensor = torch.empty(max_id + 1, dtype=torch.long, device=device)

        for bio_id, coarse_id in bio_id_to_coarse_label_id.items():
            self.mapping_tensor[bio_id] = coarse_id

    def generalize_labels(self, labels):
        return self.mapping_tensor[labels]
