import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

EXPERIMENTS_PATH = Path("experiments.csv")


def load_experiments() -> pd.DataFrame:
    """Load experiments from CSV, or return an empty DataFrame with the right columns."""
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
    NOTE: device must be one of: auto, cpu, gpu (matches runner.py argparse).
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

    if result.returncode != 0:
        output.append(f"\n[EXIT CODE: {result.returncode}]")

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

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────

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

# IMPORTANT: choices must match runner.py argparse: auto, cpu, gpu
device_choice = st.sidebar.selectbox(
    "Device",
    options=["auto", "gpu", "cpu"],
    index=0,
    help="For int8/int4, GPU (auto) is recommended.",
)

device_to_use = device_choice

# Your pipeline: bitsandbytes 4/8-bit needs GPU; force auto if CPU selected
if quantization in ("int8", "int4") and device_choice == "cpu":
    st.sidebar.warning(
        "int8/int4 on CPU is not supported in this setup – using 'auto' (GPU) instead."
    )
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

# ──────────────────────────────────────────────────────────────────────────────
# Experiments table + filters
# ──────────────────────────────────────────────────────────────────────────────

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
    # Charts: before vs after quantization (CUDA runs only)
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### Charts")

    if not df_filt.empty:
        import altair as alt

        # Only CUDA runs (auto/gpu in runner show up as 'cuda' in CSV)
        df_cuda = df_filt[df_filt["device"] == "cuda"].copy()

        if df_cuda.empty:
            st.info("No CUDA runs found for the current filters.")
        else:
            # Map quantization -> stage: fp32 = before, int8/int4 = after
            df_cuda["quant_stage"] = df_cuda["quantization"].apply(
                lambda q: "before" if q == "fp32" else "after"
            )

            # Average over multiple runs of the same (task, model, stage)
            def make_stage_df(metric_col: str) -> pd.DataFrame:
                return (
                    df_cuda.groupby(
                        ["task", "model_name", "quant_stage"], as_index=False
                    )[metric_col]
                    .mean()
                )

            acc_stage = make_stage_df("accuracy")
            lat_stage = make_stage_df("avg_latency_sec")
            co2_stage = make_stage_df("emissions_kg")

            # Helper to build a facet chart: one panel per task,
            # line connecting "before" -> "after" for each model.
            def stage_chart(data, y_field, y_title, y_format):
                base = (
                    alt.Chart(data)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "quant_stage:N",
                            title="Quantization Stage",
                            sort=["before", "after"],
                        ),
                        y=alt.Y(
                            f"{y_field}:Q",
                            title=y_title,
                            axis=alt.Axis(format=y_format),
                        ),
                        color=alt.Color("model_name:N", title="Model"),
                        tooltip=[
                            alt.Tooltip("task:N", title="Task"),
                            alt.Tooltip("model_name:N", title="Model"),
                            alt.Tooltip("quant_stage:N", title="Stage"),
                            alt.Tooltip(
                                f"{y_field}:Q", title=y_title, format=y_format
                            ),
                        ],
                    )
                    .properties(width=220, height=220)
                )
                return base.facet(column=alt.Column("task:N", title="Task"))

            st.markdown("#### Before vs After Quantization (CUDA runs)")

            st.markdown("**Accuracy: Before vs After Quantization**")
            acc_chart = stage_chart(
                acc_stage, "accuracy", "Accuracy", ".4f"
            )  # 4 decimal places
            st.altair_chart(acc_chart, use_container_width=True)

            st.markdown("**Latency: Before vs After Quantization**")
            lat_chart = stage_chart(
                lat_stage,
                "avg_latency_sec",
                "Avg Latency (s)",
                ".6f",  # higher precision
            )
            st.altair_chart(lat_chart, use_container_width=True)

            st.markdown("**Emissions: Before vs After Quantization**")
            co2_chart = stage_chart(
                co2_stage,
                "emissions_kg",
                "CO₂ Emissions (kg)",
                ".8f",  # very fine precision
            )
            st.altair_chart(co2_chart, use_container_width=True)
    else:
        st.info("No rows match the selected filters.")
