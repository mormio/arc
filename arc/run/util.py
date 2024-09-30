import re


def extract_reasoning_from_text(text):
    pattern = r"<reasoning>(.*?)</reasoning>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_code_from_text(text):
    if "</reasoning>" in text:
        text = text[text.find("</reasoning>") :]
    pattern = r"```(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def filter_by_binary(items, binary_filter):
    """For example, good for getting the functions predicted by the resnet."""
    if len(items) != len(binary_filter):
        raise ValueError("Lists must have the same length")

    return [item for item, keep in zip(items, binary_filter) if keep]


def aggregate_problem_predictions(preds, method=all):
    """Aggregates across all the predictions for the examples belonging to a single problem."""

    assert method in [any, all], "aggregation must be either any or or"
    for pid, pred_list in preds.items():
        agg = [
            int(method(row[i] for row in pred_list))
            for i in range(len(pred_list[0]))
        ]
        preds[pid] = agg

    return preds
