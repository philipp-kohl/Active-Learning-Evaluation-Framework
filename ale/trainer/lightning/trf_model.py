from typing import List, Dict

import torch
import torchmetrics
from lightning import LightningModule
from torch import optim, softmax
from torchmetrics import Metric
from transformers import AutoModelForTokenClassification

from ale.trainer.lightning.utils import derive_labels


def is_valid_for_prog_bar(metric_name: str):
    return "f1_macro" in metric_name.lower() or "f1_micro" in metric_name.lower()


class TransformerLightning(LightningModule):
    def __init__(self, model_name: str, labels: List[str], learn_rate: float, weight_decay: float,
                 ignore_labels: List[str] = None):
        super().__init__()
        self.save_hyperparameters()
        if ignore_labels is None:
            ignore_labels = []

        self.id2label, self.label2id = derive_labels(labels)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(self.id2label),
                                                                     id2label=self.id2label, label2id=self.label2id)
        self.learn_rate = learn_rate
        self.num_labels = len(self.id2label)

        self.f1_macro_per_label = torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                       average=None,
                                                       ignore_index=-1)
        self.val_metrics = {
            "val_precision_micro": torchmetrics.Precision(task="multiclass", num_classes=self.num_labels,
                                                          average='micro', ignore_index=-1),
            "val_recall_micro": torchmetrics.Recall(task="multiclass", num_classes=self.num_labels,
                                                    average='micro',
                                                    ignore_index=-1),
            "val_f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                 average='micro',
                                                 ignore_index=-1),
            "val_precision_macro": torchmetrics.Precision(task="multiclass", num_classes=self.num_labels,
                                                          average='macro', ignore_index=-1),
            "val_recall_macro": torchmetrics.Recall(task="multiclass", num_classes=self.num_labels,
                                                    average='macro',
                                                    ignore_index=-1),
            "val_f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                 average='macro',
                                                 ignore_index=-1)
        }
        self.train_metrics = {
            "train_precision_micro": torchmetrics.Precision(task="multiclass", num_classes=self.num_labels,
                                                            average='micro', ignore_index=-1),
            "train_recall_micro": torchmetrics.Recall(task="multiclass", num_classes=self.num_labels,
                                                      average='micro',
                                                      ignore_index=-1),
            "train_f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                   average='micro',
                                                   ignore_index=-1),
            "train_precision_macro": torchmetrics.Precision(task="multiclass", num_classes=self.num_labels,
                                                            average='macro', ignore_index=-1),
            "train_recall_macro": torchmetrics.Recall(task="multiclass", num_classes=self.num_labels,
                                                      average='macro',
                                                      ignore_index=-1),
            "train_f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                   average='macro',
                                                   ignore_index=-1)}
        self.test_metrics = {
            "test_precision_micro": torchmetrics.Precision(task="multiclass", num_classes=self.num_labels,
                                                           average='micro', ignore_index=-1),
            "test_recall_micro": torchmetrics.Recall(task="multiclass", num_classes=self.num_labels,
                                                     average='micro',
                                                     ignore_index=-1),
            "test_f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                  average='micro',
                                                  ignore_index=-1),
            "test_precision_macro": torchmetrics.Precision(task="multiclass", num_classes=self.num_labels,
                                                           average='macro', ignore_index=-1),
            "test_recall_macro": torchmetrics.Recall(task="multiclass", num_classes=self.num_labels,
                                                     average='macro',
                                                     ignore_index=-1),
            "test_f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels,
                                                  average='macro',
                                                  ignore_index=-1)}
        self.ignore_labels = ignore_labels
        self.weight_decay = weight_decay

    def on_fit_start(self):
        self.model = self.model.to(self.device)
        for metric_name, metric in self.val_metrics.items():
            self.val_metrics[metric_name] = metric.to(self.device)
        for metric_name, metric in self.train_metrics.items():
            self.train_metrics[metric_name] = metric.to(self.device)
        for metric_name, metric in self.test_metrics.items():
            self.test_metrics[metric_name] = metric.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.evaluate(batch, outputs, self.train_metrics)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.evaluate(batch, outputs, self.val_metrics)
        self.log_dict({'val_loss': loss})

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.evaluate(batch, outputs, self.test_metrics)
        self.log_dict({'test_loss': loss})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        logits = outputs.logits

        mask = batch['attention_mask']
        softmax_logits = softmax(logits, dim=-1)
        class_predictions = torch.argmax(softmax_logits, dim=-1)
        class_predictions_numpy = class_predictions.cpu().numpy() if class_predictions.is_cuda else class_predictions.numpy()
        class_predictions_numpy = self.apply_mask(mask, class_predictions_numpy)

        token_labels = []
        for sequence in class_predictions_numpy:
            sequence_labels = [self.id2label[index] for index in sequence]
            token_labels.append(sequence_labels)
        token_labels = self.apply_mask(mask, token_labels)

        confidences = []
        confidences_numpy = softmax_logits.cpu().numpy() if softmax_logits.is_cuda else softmax_logits.numpy()
        for conf_batch in confidences_numpy:
            single_sequence = []
            for token in conf_batch:
                single_sequence.append(
                    {self.id2label[idx]: token_confidence for idx, token_confidence in enumerate(token)})
            confidences.append(single_sequence)
        confidences = self.apply_mask(mask, confidences)

        return {'tokens': batch['token_text'], 'token_labels': token_labels,
                'text': batch['text'], 'offset_mapping': batch['offset_mapping'],
                'confidences': confidences}

    def apply_mask(self, mask, value):
        result = []
        for m, v in zip(mask, value):
            seq = []
            for m1, v1 in zip(m, v):
                if m1 == 1:
                    seq.append(v1)
            result.append(seq)

        return result

    def on_validation_epoch_end(self):
        for metric_name, metric in self.val_metrics.items():
            self.log(metric_name, metric.compute(), prog_bar=is_valid_for_prog_bar(metric_name))
        for idx, score in enumerate(self.f1_macro_per_label.compute()):
            self.log(f"val_f1_macro_${self.id2label[idx]}", score, prog_bar=False)

    def on_train_epoch_end(self):
        for metric_name, metric in self.train_metrics.items():
            self.log(metric_name, metric.compute(), prog_bar=is_valid_for_prog_bar(metric_name))

    def on_test_epoch_end(self):
        for metric_name, metric in self.test_metrics.items():
            self.log(metric_name, metric.compute(), prog_bar=is_valid_for_prog_bar(metric_name))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learn_rate, weight_decay=self.weight_decay)
        return optimizer

    def evaluate(self, batch, outputs, metrics: Dict[str, Metric]):
        mask = batch["attention_mask"]
        gold_labels = batch["labels"]
        prediction_labels = torch.argmax(outputs.logits, dim=-1)
        mask_flat = mask.view(-1)
        gold_labels_flat = gold_labels.view(-1)
        prediction_labels_flat = prediction_labels.view(-1)
        # Apply mask: Set predictions to -1 where mask is 0 (padded)
        prediction_labels_flat = torch.where(mask_flat == 1, prediction_labels_flat,
                                             torch.tensor(-1, device=self.device))
        gold_labels_flat = torch.where(mask_flat == 1, gold_labels_flat, torch.tensor(-1, device=self.device))
        for l in self.ignore_labels:
            label_idx = self.label2id[l]
            prediction_labels_flat = torch.where(prediction_labels_flat != label_idx,
                                                 prediction_labels_flat,
                                                 torch.tensor(-1, device=self.device))
            gold_labels_flat = torch.where(gold_labels_flat != label_idx, gold_labels_flat,
                                           torch.tensor(-1, device=self.device))
        # Filter out the ignored indices (-1) before passing them to the metrics
        valid_indices = gold_labels_flat != -1  # Assuming -1 is used to mark padded or ignored labels
        valid_gold_labels = gold_labels_flat[valid_indices]
        valid_prediction_labels = prediction_labels_flat[valid_indices]
        # Update metrics with filtered valid labels and predictions
        for metric_name, metric in metrics.items():
            metric(valid_prediction_labels, valid_gold_labels)
        self.f1_macro_per_label(valid_prediction_labels, valid_gold_labels)
