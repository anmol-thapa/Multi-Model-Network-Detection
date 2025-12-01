import json
import random
import subprocess
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")
MODELS_PATH = ARTIFACTS_DIR / "models.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEFAULT_DATA_PATH = Path("UNSW-NB15/UNSW_NB15_training-set.csv")


def ensure_artifacts():
    if MODELS_PATH.exists() and METADATA_PATH.exists():
        return
    data_path = Path(st.secrets["DATA_PATH"]) if "DATA_PATH" in st.secrets else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Artifacts missing and dataset not found at {data_path}. "
            "Upload the UNSW_NB15_training-set.csv file or set st.secrets['DATA_PATH']."
        )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable,
        "train_model.py",
        "--data-path",
        str(data_path),
        "--target",
        "label",
        "--live-features",
    ]
    with st.spinner("Artifacts missing — training models (first run only)..."):
        subprocess.run(train_cmd, check=True)


@st.cache_resource
def load_artifacts():
    ensure_artifacts()
    models = joblib.load(MODELS_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return models, metadata


def build_input_defaults(metadata):
    defaults = {}
    for col, stats in metadata["numeric_stats"].items():
        defaults[col] = stats["median"]
    for col, val in metadata["categorical_defaults"].items():
        defaults[col] = val
    return defaults


def main():
    st.set_page_config(page_title="UNSW-NB15 Intrusion Detector", page_icon="🛡️", layout="wide")
    st.title("UNSW-NB15 Multi-Model Predictor")
    st.write("Adjust key network flow features and see how the model classifies the flow.")

    try:
        models, metadata = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Train artifacts with: `python train_model.py --data-path /path/to/UNSW_NB15_training-set.csv`")
        return

    feature_columns = metadata["feature_columns"]
    numeric_stats = metadata["numeric_stats"]
    categorical_levels = metadata["categorical_levels"]
    top_features = metadata.get("top_features", [])
    feature_descriptions = metadata.get("feature_descriptions", {})
    class_examples = metadata.get("class_examples", {})
    class_labels = metadata.get("class_labels", {})
    model_order = ["RandomForest", "LogisticRegression", "XGBoost"]

    defaults = build_input_defaults(metadata)
    if "inputs" not in st.session_state:
        st.session_state["inputs"] = defaults.copy()

    # Centered content with balanced columns
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.subheader("Try preset examples")
        if "0" in class_examples:
            if st.button("Load benign (normal)"):
                choice = random.choice(class_examples["0"])
                for k, v in choice.items():
                    if k in numeric_stats:
                        st.session_state["inputs"][k] = float(v)
                    else:
                        st.session_state["inputs"][k] = v
        if "1" in class_examples:
            if st.button("Load attack"):
                choice = random.choice(class_examples["1"])
                for k, v in choice.items():
                    if k in numeric_stats:
                        st.session_state["inputs"][k] = float(v)
                    else:
                        st.session_state["inputs"][k] = v

        st.subheader("Inputs")
        cols = st.columns(2)
        input_data = st.session_state["inputs"]
        for idx, feat in enumerate(top_features):
            col = cols[idx % 2]
            help_txt = feature_descriptions.get(
                feat,
                "Important to the model. Higher or lower values can shift the risk score.",
            )
            if feat in categorical_levels:
                options = categorical_levels[feat]
                current = input_data.get(feat, options[0] if options else None)
                chosen = col.selectbox(f"{feat}", options, index=options.index(current) if current in options else 0, help=help_txt)
                input_data[feat] = chosen
            elif feat in numeric_stats:
                stats = numeric_stats[feat]
                lo = float(stats.get("p1", stats["min"]))
                hi = float(stats.get("p99", stats["max"]))
                if lo == hi:
                    hi = lo + 1.0
                median = float(stats["median"])
                step = max((hi - lo) / 100.0, 0.01)
                current = float(input_data.get(feat, median))
                val = col.slider(f"{feat}", lo, hi, current, step=step, help=help_txt)
                input_data[feat] = float(val)

    # Build full row matching training columns, filling non-top features with defaults
    row = {col: input_data.get(col, defaults.get(col, 0.0)) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)

    with col_right:
        st.subheader("Model predictions")
        # Two-row grid on the right: top RF, bottom LR; XGBoost if available
        containers = []
        for i in range(3):
            containers.append(st.container())

        def render_model(idx: int, name: str):
            if name not in models:
                return
            pipe = models[name]
            proba = pipe.predict_proba(df)[0]
            pred_idx = int(np.argmax(proba))
            pred_label = pipe.classes_[pred_idx]
            display_label = class_labels.get(str(pred_label), str(pred_label))
            containers[idx].markdown(f"### {name}")
            containers[idx].write(f"Prediction: **{display_label}**")
            chart_df = pd.DataFrame(
                {
                    "class": [class_labels.get(str(cls), str(cls)) for cls in pipe.classes_],
                    "probability": proba,
                }
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                    y=alt.Y("class:N", sort="-x", title=None),
                    tooltip=["class", alt.Tooltip("probability:Q", format=".3f")],
                )
                .properties(height=120)
                .configure_view(stroke=None)
            )
            containers[idx].altair_chart(chart, use_container_width=True)

        # Top: RandomForest if present
        render_model(0, "RandomForest")
        # Middle: LogisticRegression
        render_model(1, "LogisticRegression")
        # Bottom: XGBoost if trained
        render_model(2, "XGBoost")

    st.caption("Compare how each model responds to the same inputs.")

    # Feature importance from the best model (or first available)
    def aggregate_importances(feature_names, importances):
        grouped = {}
        for name, imp in zip(feature_names, importances):
            raw = name.split("__", 1)[-1]
            raw_base = raw.split("_", 1)[0]
            grouped[raw_base] = grouped.get(raw_base, 0.0) + float(imp)
        return sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)

    best_model_name = metadata.get("best_model")
    model_for_importance = models.get(best_model_name) or next(iter(models.values()))
    preprocess = model_for_importance.named_steps["preprocess"]
    model_core = model_for_importance.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    if hasattr(model_core, "feature_importances_"):
        importances = model_core.feature_importances_
    elif hasattr(model_core, "coef_"):
        importances = np.abs(model_core.coef_).ravel()
    else:
        importances = np.zeros(len(feature_names))

    aggregated = aggregate_importances(feature_names, importances)[:15]
    if aggregated:
        st.subheader("Top feature importance (best model)")
        imp_df = pd.DataFrame(aggregated, columns=["feature", "importance"])
        imp_chart = (
            alt.Chart(imp_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(height=300)
            .configure_view(stroke=None)
        )
        st.altair_chart(imp_chart, use_container_width=True)


if __name__ == "__main__":
    main()
