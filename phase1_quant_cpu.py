from pathlib import Path
import os
os.environ["CODECARBON_LOG_LEVEL"] = "ERROR"

from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Create emissions folder
Path("emissions").mkdir(parents=True, exist_ok=True)

with EmissionsTracker(project_name="phase1-quant-cpu", output_dir="./emissions") as tracker:
    model_id = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_id)

    # Load model on CPU
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")

    # Apply dynamic quantization (only works on CPU)
    quant_model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    ).to("cpu")

    # Summarization pipeline on CPU
    summarizer = pipeline("summarization", model=quant_model, tokenizer=tok, device=-1)

    # --- good generation settings for T5 summarization ---
gen_kwargs = dict(
    max_new_tokens=48,          # keep short; remove max_length to avoid warnings
    num_beams=4,                # small beam search for stability
    no_repeat_ngram_size=3,     # avoids repetition
    length_penalty=1.0,         # neutral
)

# Prefix inputs with "summarize: "
inputs = [
    "summarize: AI models consume significant energy; quantization reduces computation.",
    "summarize: Harika compares carbon emissions before and after optimization.",
    "summarize: Sustainable AI focuses on reducing GPU energy without hurting accuracy."
]

outputs = summarizer(inputs, truncation=True, **gen_kwargs)
for i, out in enumerate(outputs, 1):
    print(f"\n--- Sample {i} ---")
    print("SUMMARY:", out["summary_text"])

