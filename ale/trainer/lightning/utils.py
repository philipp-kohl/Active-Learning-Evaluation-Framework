def derive_labels(labels):
    def enumerate_v2(xs, start=0, step=1):
        for x in xs:
            yield start, x
            start += step

    id2label = {0: "O"}
    for idx, label in enumerate_v2(labels, start=1, step=2):
        id2label[idx] = f"B-{label}"
        id2label[idx + 1] = f"I-{label}"
    label2id = {value: key for key, value in id2label.items()}
    return id2label, label2id
