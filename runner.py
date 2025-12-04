# runner.py

import argparse
import os
import time

import torch
import pandas as pd
from codecarbon import EmissionsTracker
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)

from utils.datasets import load_summarization_data, load_qa_data, load_reasoning_data
# ^ make sure this import includes load_reasoning_data now

from utils.metrics import rouge_like, f1_for_qa


# -------------------------
# 1. Device resolution
# -------------------------

def resolve_device(device_arg: str) -> str:
    """
    device_arg: 'auto', 'cpu', or 'gpu'
    returns: 'cpu' or 'cuda'
    """
    device_arg = device_arg.lower()

    if device_arg == "cpu":
        return "cpu"

    if device_arg == "gpu":
        if torch.cuda.is_available():
            print("[INFO] Using GPU (cuda).")
            return "cuda"
        else:
            print("[WARN] GPU requested but no CUDA available. Falling back to CPU.")
            return "cpu"

    # 'auto' or anything else → pick the best available
    if torch.cuda.is_available():
        print("[INFO] Auto device: using GPU (cuda).")
        return "cuda"
    else:
        print("[INFO] Auto device: no GPU available, using CPU.")
        return "cpu"


# -------------------------
# 2. Model loader with quantization
# -------------------------

def load_model_and_tokenizer(task: str, model_name: str, quantization: str, device: str):
    """
    task: 'summarization' or 'qa'
    quantization: 'fp32', 'int8', 'int4'
    device: 'cpu' or 'cuda'
    """
    if task in ("summarization", "reasoning"):
        ModelCls = AutoModelForSeq2SeqLM
    elif task == "qa":
        ModelCls = AutoModelForQuestionAnswering
    else:
        raise ValueError(f"Unsupported task: {task}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # kwargs for quantization
    kwargs = {}

    if quantization in ("int8", "int4"):
        # bitsandbytes-style quantization, GPU only
        kwargs["device_map"] = "auto"
        if quantization == "int8":
            kwargs["load_in_8bit"] = True
        elif quantization == "int4":
            kwargs["load_in_4bit"] = True

        print(f"[INFO] Loading {model_name} with {quantization} quantization.")
        model = ModelCls.from_pretrained(model_name, **kwargs)
    else:
        # fp32
        print(f"[INFO] Loading {model_name} in full precision on {device}.")
        model = ModelCls.from_pretrained(model_name)
        model.to(device)

    model.eval()
    return tokenizer, model


# -------------------------
# 3. Task-specific experiment runners
# -------------------------

def run_summarization_experiment(model_name: str, quantization: str, device: str):
    data = load_summarization_data()
    tokenizer, model = load_model_and_tokenizer("summarization", model_name, quantization, device)

    scores = []
    latencies = []

    for example in data:
        inputs = tokenizer(example["input"], return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        latency = time.time() - start

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        score = rouge_like(pred, example["reference"])

        scores.append(score)
        latencies.append(latency)

    avg_score = sum(scores) / len(scores)
    avg_latency = sum(latencies) / len(latencies)
    return avg_score, avg_latency


def run_qa_experiment(model_name: str, quantization: str, device: str):
    data = load_qa_data()
    tokenizer, model = load_model_and_tokenizer("qa", model_name, quantization, device)

    scores = []
    latencies = []

    for example in data:
        inputs = tokenizer(
            example["question"],
            example["context"],
            return_tensors="pt",
        ).to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        latency = time.time() - start

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        start_index = torch.argmax(start_logits).item()
        end_index = torch.argmax(end_logits).item()

        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index + 1])

        score = f1_for_qa(pred_answer, example["answer"])

        scores.append(score)
        latencies.append(latency)

    avg_score = sum(scores) / len(scores)
    avg_latency = sum(latencies) / len(latencies)
    return avg_score, avg_latency


# -------------------------
# 4. CodeCarbon wrapper + CSV logging
# -------------------------

def run_with_carbon(task: str, model_name: str, quantization: str, device: str):
    os.makedirs("emissions", exist_ok=True)

    tracker = EmissionsTracker(
        project_name="sustainable-ai",
        output_dir="emissions",
        save_to_file=True,
    )

    tracker.start()
    start_time = time.time()

    if task == "summarization":
        accuracy, avg_latency = run_summarization_experiment(model_name, quantization, device)
    elif task == "qa":
        accuracy, avg_latency = run_qa_experiment(model_name, quantization, device)
    elif task == "reasoning":
        accuracy, avg_latency = run_reasoning_experiment(model_name, quantization, device)
    else:
        raise ValueError(f"Unsupported task: {task}")

    emissions_kg = tracker.stop()
    end_time = time.time()
    total_time = end_time - start_time

    result = {
        "task": task,
        "model_name": model_name,
        "quantization": quantization,
        "device": device,
        "accuracy": accuracy,
        "avg_latency_sec": avg_latency,
        "total_time_sec": total_time,
        "emissions_kg": emissions_kg,
    }

    print("Experiment result:", result)

    df = pd.DataFrame([result])
    if os.path.exists("experiments.csv"):
        df.to_csv("experiments.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("experiments.csv", index=False)


# -------------------------
# 5. CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["summarization", "qa", "reasoning"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="fp32", choices=["fp32", "int8", "int4"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device preference: 'auto' (default), 'cpu', or 'gpu'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    actual_device = resolve_device(args.device)  # 'cpu' or 'cuda'

    run_with_carbon(
        task=args.task,
        model_name=args.model_name,
        quantization=args.quantization,
        device=actual_device,
    )
def run_reasoning_experiment(model_name: str, quantization: str, device: str):
    data = load_reasoning_data()
    tokenizer, model = load_model_and_tokenizer("reasoning", model_name, quantization, device)

    scores = []
    latencies = []

    for example in data:
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        latency = time.time() - start

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        score = rouge_like(pred, example["reference"])

        scores.append(score)
        latencies.append(latency)

    avg_score = sum(scores) / len(scores)
    avg_latency = sum(latencies) / len(latencies)
    return avg_score, avg_latency


if __name__ == "__main__":
    main()
