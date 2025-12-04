from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from torch.ao.quantization import quantize_dynamic
from codecarbon import EmissionsTracker
from pathlib import Path
import torch

Path("emissions").mkdir(parents=True, exist_ok=True)

with EmissionsTracker(project_name="phase1-quant-gpu", output_dir="./emissions") as tracker:
    model_id = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_id)

    # Load FP32 model on CPU for quantization
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")
    quant_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)

    summarizer = pipeline("summarization", model=quant_model, tokenizer=tok, framework="pt")

    inputs = [
        "AI models consume significant energy; quantization reduces computation.",
        "Harika compares carbon emissions before and after optimization.",
        "Sustainable AI focuses on reducing GPU energy without hurting accuracy."
    ]

    outputs = summarizer(inputs, max_length=60, min_length=10, truncation=True)
    for i, out in enumerate(outputs, 1):
        print(f"\n--- Sample {i} ---")
        print("SUMMARY:", out["summary_text"])

