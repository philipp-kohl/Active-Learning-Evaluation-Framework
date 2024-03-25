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
