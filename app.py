import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

EXPERIMENTS_PATH = Path("experiments.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_experiments() -> pd.DataFrame:
    """Load experiments.csv or return an empty DataFrame with the right columns."""
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
    NOTE: runner.py expects device in {auto, cpu, gpu}.
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

    output_parts = ["COMMAND: " + " ".join(cmd)]

    if result.stdout:
        output_parts.append("\nSTDOUT:\n" + result.stdout)
    if result.stderr:
        output_parts.append("\nSTDERR:\n" + result.stderr)

    if result.returncode != 0:
        output_parts.append(f"\n[EXIT CODE: {result.returncode}]")

    return "\n".join(output_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit layout
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Greener AI – Experiments", layout="wide")

st.title("🌿 Greener AI: Model · Quantization · Emissions Explorer")

st.markdown(
    """
This tool has **two views**:

- **Run Experiment** – launch a new run on the Azure VM and append results to `experiments.csv`.
- **Results & Analysis** – explore recorded runs and compare **before vs after quantization**
  for **accuracy**, **latency**, and **CO₂ emissions**.
"""
)

# Top-level navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("View", ["Run Experiment", "Results & Analysis"])

# ──────────────────────────────────────────────────────────────────────────────
# VIEW 1: Run Experiment
# ──────────────────────────────────────────────────────────────────────────────

if mode == "Run Experiment":
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

    # IMPORTANT: device choices match argparse {auto, cpu, gpu}
    device_choice = st.sidebar.selectbox(
        "Requested device",
        options=["auto", "gpu", "cpu"],
        index=0,
        help="For int8/int4, GPU (auto or gpu) is recommended.",
    )

    device_to_use = device_choice

    # BitsAndBytes int8/int4 requires GPU in your setup
    if quantization in ("int8", "int4") and device_choice == "cpu":
        st.sidebar.warning(
            "int8/int4 on CPU is not supported in this setup – using 'auto' (GPU) instead."
        )
        device_to_use = "auto"

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run experiment")

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

    st.markdown("## 📄 All recorded experiments")
    df = load_experiments()
    if df.empty:
        st.info("No experiments recorded yet. Run one from the sidebar to get started.")
    else:
        st.dataframe(df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# VIEW 2: Results & Analysis
# ──────────────────────────────────────────────────────────────────────────────
else:
    df = load_experiments()

    st.subheader("📊 Results & Analysis")

    if df.empty:
        st.info("No experiments recorded yet. Go to **Run Experiment** to create some.")
    else:
        # Sidebar filters for RESULTS view
        st.sidebar.header("Results filters")

        # ── Task filter with "(all tasks)" option ──
        tasks_available = sorted(df["task"].unique().tolist())
        task_options = ["(all tasks)"] + tasks_available
        selected_task = st.sidebar.selectbox(
            "Task",
            options=task_options,
            index=0,
        )

        # Start from full DF, then filter if a specific task is chosen
        df_task = df.copy()
        if selected_task != "(all tasks)":
            df_task = df_task[df_task["task"] == selected_task]

        # Only CUDA?
        only_cuda = st.sidebar.checkbox("Only include GPU (cuda) runs", value=True)
        if only_cuda:
            df_task = df_task[df_task["device"] == "cuda"]

        # Model filter (multi-select)
        if df_task.empty:
            models_available = []
        else:
            models_available = sorted(df_task["model_name"].unique().tolist())

        selected_models = st.sidebar.multiselect(
            "Models to include",
            options=models_available,
            default=models_available,
        )

        if selected_models:
            df_task = df_task[df_task["model_name"].isin(selected_models)]

        # MAIN AREA
        st.markdown("### Raw experiments table (after filters)")
        if df_task.empty:
            st.warning(
                "No rows left after applying filters. "
                "Try disabling 'Only include GPU' or selecting more tasks/models."
            )
        else:
            st.dataframe(df_task, use_container_width=True)

            # Add stage: before (fp32) vs after (int8/int4)
            df_task["quant_stage"] = df_task["quantization"].apply(
                lambda q: "before" if q == "fp32" else "after"
            )

            # Aggregate over multiple runs of same (model, stage)
            agg = (
                df_task.groupby(
                    ["model_name", "quant_stage"], as_index=False
                )
                .agg(
                    accuracy=("accuracy", "mean"),
                    avg_latency_sec=("avg_latency_sec", "mean"),
                    emissions_kg=("emissions_kg", "mean"),
                )
            )

            # Friendly label for title
            if selected_task == "(all tasks)":
                task_label = "all tasks"
            else:
                task_label = selected_task

            st.markdown(
                f"### Before vs After Quantization – **{task_label}**"
            )
            st.caption(
                "Each chart shows two lines per model: "
                "`before` = fp32 baseline, `after` = quantized (int8/int4). "
                "Values are averaged over all runs matching the filters."
            )

            def plot_metric(metric: str, title: str, y_title: str, fmt: str):
                return (
                    alt.Chart(agg)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "model_name:N",
                            title="Model",
                            sort=sorted(agg["model_name"].unique().tolist()),
                            axis=alt.Axis(labelAngle=0),
                        ),
                        y=alt.Y(
                            f"{metric}:Q",
                            title=y_title,
                            axis=alt.Axis(format=fmt),
                        ),
                        color=alt.Color(
                            "quant_stage:N",
                            title="Stage",
                            sort=["before", "after"],
                        ),
                        tooltip=[
                            alt.Tooltip("model_name:N", title="Model"),
                            alt.Tooltip("quant_stage:N", title="Stage"),
                            alt.Tooltip(f"{metric}:Q", title=y_title, format=fmt),
                        ],
                    )
                    .properties(width=500, height=300, title=title)
                )

            # Accuracy chart
            acc_chart = plot_metric(
                "accuracy",
                "Accuracy: Before vs After Quantization",
                "Accuracy",
                ".4f",
            )
            st.altair_chart(acc_chart, use_container_width=True)

            # Latency chart
            lat_chart = plot_metric(
                "avg_latency_sec",
                "Latency: Before vs After Quantization",
                "Avg Latency (s)",
                ".6f",
            )
            st.altair_chart(lat_chart, use_container_width=True)

            # Emissions chart
            co2_chart = plot_metric(
                "emissions_kg",
                "CO₂ Emissions: Before vs After Quantization",
                "CO₂ Emissions (kg)",
                ".8f",
            )
            st.altair_chart(co2_chart, use_container_width=True)
