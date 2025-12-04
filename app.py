import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


EXPERIMENTS_PATH = Path("experiments.csv")


def load_experiments() -> pd.DataFrame:
    if not EXPERIMENTS_PATH.exists():
        return pd.DataFrame(
            columns=[
                "task",
                "model_name",
                "quantization",
                "device",
                "accuracy",
                "avg_latency_sec",
                "total_time_sec",
                "emissions_kg",
            ]
        )
    return pd.read_csv(EXPERIMENTS_PATH)


def run_experiment(task: str, model_name: str, quantization: str, device: str) -> str:
    """
    Run runner.py with the selected options and return the combined stdout/stderr.
    """
    cmd = [
        sys.executable,
        "runner.py",
        "--task",
        task,
        "--model_name",
        model_name,
        "--quantization",
        quantization,
        "--device",
        device,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    output = []
    output.append("COMMAND: " + " ".join(cmd))
    if result.stdout:
        output.append("\nSTDOUT:\n" + result.stdout)
    if result.stderr:
        output.append("\nSTDERR:\n" + result.stderr)

    return "\n".join(output)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Greener AI – Experiments",
    layout="wide",
)

st.title("🌿 Greener AI: Model · Quantization · Emissions Explorer")

st.markdown(
    """
Use this dashboard to:

1. **Choose a task, model, quantization, and device**
2. **Run an experiment** on the Azure VM (with CodeCarbon)
3. **Compare accuracy, latency, and CO₂ emissions** across runs
"""
)

df = load_experiments()

# Sidebar controls
st.sidebar.header("Experiment configuration")

task = st.sidebar.selectbox(
    "Task",
    options=["summarization", "qa", "reasoning"],
    index=0,
)

if task == "summarization":
    model_options = [
        "google/flan-t5-small",
        "google/flan-t5-base",
    ]
elif task == "qa":
    model_options = [
        "distilbert-base-uncased-distilled-squad",
    ]
else:  # reasoning
    model_options = [
        "google/flan-t5-base",
    ]

model_name = st.sidebar.selectbox("Model", options=model_options)

quantization = st.sidebar.selectbox(
    "Quantization",
    options=["fp32", "int8", "int4"],
    index=0,
)

device_choice = st.sidebar.selectbox(
    "Device",
    options=["auto", "cuda", "cpu"],
    index=0,
    help="For int8/int4, GPU (auto/cuda) is recommended.",
)

# Guard: int8/int4 on CPU is not supported in your pipeline
device_to_use = device_choice
if quantization in ("int8", "int4") and device_choice == "cpu":
    st.sidebar.warning("int8/int4 on CPU not supported – using 'auto' (GPU) instead.")
    device_to_use = "auto"

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run experiment")

# ──────────────────────────────────────────────────────────────────────────────
# Run section
# ──────────────────────────────────────────────────────────────────────────────

if run_button:
    with st.status("Running experiment...", expanded=True) as status:
        st.write(
            f"Task: `{task}`, Model: `{model_name}`, "
            f"Quantization: `{quantization}`, Device: `{device_to_use}`"
        )
        output = run_experiment(task, model_name, quantization, device_to_use)
        status.update(label="Experiment finished", state="complete")

    st.subheader("Raw runner output")
    st.code(output, language="bash")

    # Reload experiments after new run
    df = load_experiments()

st.markdown("## 📊 Current experiments")

if df.empty:
    st.info("No experiments recorded yet. Run one from the sidebar to get started.")
else:
    st.dataframe(df, use_container_width=True)

    st.markdown("### Filtered view")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        filt_task = st.selectbox(
            "Filter by task",
            options=["(all)"] + sorted(df["task"].unique().tolist()),
            index=0,
        )
    with col_b:
        filt_model = st.selectbox(
            "Filter by model",
            options=["(all)"] + sorted(df["model_name"].unique().tolist()),
            index=0,
        )
    with col_c:
        filt_quant = st.selectbox(
            "Filter by quantization",
            options=["(all)"] + sorted(df["quantization"].unique().tolist()),
            index=0,
        )

    df_filt = df.copy()
    if filt_task != "(all)":
        df_filt = df_filt[df_filt["task"] == filt_task]
    if filt_model != "(all)":
        df_filt = df_filt[df_filt["model_name"] == filt_model]
    if filt_quant != "(all)":
        df_filt = df_filt[df_filt["quantization"] == filt_quant]

    st.dataframe(df_filt, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # Charts: accuracy, latency, emissions vs model & quantization
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### Charts")

    if not df_filt.empty:
        import altair as alt

        base = alt.Chart(df_filt).encode(
            x=alt.X("model_name:N", title="Model"),
            color=alt.Color("quantization:N", title="Quantization"),
            column=alt.Column("task:N", title="Task"),
        )

        acc_chart = base.mark_bar().encode(
            y=alt.Y("accuracy:Q", title="Accuracy"),
        ).properties(title="Accuracy by model & quantization")

        lat_chart = base.mark_bar().encode(
            y=alt.Y("avg_latency_sec:Q", title="Avg latency (s)"),
        ).properties(title="Latency by model & quantization")

        co2_chart = base.mark_bar().encode(
            y=alt.Y("emissions_kg:Q", title="Emissions (kg CO₂)"),
        ).properties(title="Emissions by model & quantization")

        st.altair_chart(acc_chart, use_container_width=True)
        st.altair_chart(lat_chart, use_container_width=True)
        st.altair_chart(co2_chart, use_container_width=True)
    else:
        st.info("No rows match the selected filters.")
