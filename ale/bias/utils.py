from typing import Dict


def normalize_counts(counts_dict: Dict[str, float]):
    """
    Normalizes the counts in the given defaultdict by the total count of all labels.

    Parameters:
    - counts_dict (defaultdict): A defaultdict with label counts.

    Returns:
    - dict: A dictionary with the same keys as counts_dict, but with values normalized
            so that they sum to 1.
    """
    total_count = sum(counts_dict.values())
    if total_count == 0:
        return {}

    normalized_dict = {label: count / total_count for label, count in counts_dict.items()}
    return normalized_dict
