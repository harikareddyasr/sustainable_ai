from typing import Dict, Any, List
from datasets import load_dataset

# XSum:
# - We use sentence-transformers/xsum (script-free parquet)
# - It only provides 'train', so we map validation -> train[:5000]
def load_summarization_data(dataset: str = "xsum", split: str = "validation"):
    if dataset != "xsum":
        raise ValueError(f"Unsupported summarization dataset: {dataset}")

    if split == "validation":
        hf_split = "train[:5000]"
    elif split == "train":
        hf_split = "train"
    else:
        raise ValueError('For xsum mirror, split must be "train" or "validation".')

    ds = load_dataset("sentence-transformers/xsum", split=hf_split)

    def _map(ex: Dict[str, Any]):
        return {"input": ex["article"], "reference": ex["summary"]}

    return ds.map(_map, remove_columns=ds.column_names)


# SQuAD (parquet)
def load_qa_data(dataset: str = "squad", split: str = "validation"):
    if dataset != "squad":
        raise ValueError(f"Unsupported QA dataset: {dataset}")

    ds = load_dataset("squad", split=split)

    def _map(ex: Dict[str, Any]):
        answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        return {"question": ex["question"], "context": ex["context"], "answer": answer}

    return ds.map(_map, remove_columns=ds.column_names)


# ARC-Challenge (parquet)
# Your machine's format:
#   question: str
#   choices: dict with keys "text": [..], "label": [..]
#   answerKey: "A"/"B"/...
def load_reasoning_data(dataset: str = "arc_challenge", split: str = "validation"):
    if dataset != "arc_challenge":
        raise ValueError(f"Unsupported reasoning dataset: {dataset}")

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)

    def _map(ex: Dict[str, Any]):
        stem = ex["question"] if isinstance(ex.get("question"), str) else ""

        ch = ex.get("choices", {})
        labels: List[str] = ch.get("label", []) if isinstance(ch, dict) else []
        texts: List[str] = ch.get("text", []) if isinstance(ch, dict) else []

        choice_lines = "\n".join([f"{lab}) {txt}" for lab, txt in zip(labels, texts)])

        prompt = (
            f"Question: {stem}\n"
            f"Choices:\n{choice_lines}\n\n"
            f"Answer with only the letter of the correct choice."
        )

        label = (ex.get("answerKey") or "").strip().upper()
        return {"prompt": prompt, "label": label}

    return ds.map(_map, remove_columns=ds.column_names)
