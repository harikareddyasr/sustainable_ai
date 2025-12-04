def rouge_like(pred: str, ref: str) -> float:
    """
    Very rough ROUGE-style overlap measure.
    We'll swap this for proper ROUGE later if needed.
    """
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    return overlap / len(ref_tokens)


def f1_for_qa(pred: str, ref: str) -> float:
    """
    Simple token-level F1 score for QA.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set & ref_set
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
