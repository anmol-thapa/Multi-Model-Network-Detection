import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

RANDOM_STATE = 42

# Features that cannot be captured in Wireshark
DROP_ALWAYS = ["id"]
WIRESHARK_NON_SCANNABLE = ["service", "trans_depth", "response_body_len", "is_ftp_login", "is_sm_ips_ports"]


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric_features)]

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.insert(0, ("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, categorical_features, numeric_features


def aggregate_importances(feature_names: list[str], importances) -> list[tuple[str, float]]:
    grouped: dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        # ColumnTransformer prefixes with transformer name, OneHotEncoder adds category suffixes
        raw = name.split("__", 1)[-1]
        raw_base = raw.split("_", 1)[0]
        grouped[raw_base] = grouped.get(raw_base, 0.0) + float(imp)
    return sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)


def collect_metadata(df: pd.DataFrame, categorical_features: list[str], numeric_features: list[str], target: str, top_features: list[str]):
    categorical_levels = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_features}
    categorical_defaults = {col: (levels[0] if levels else None) for col, levels in categorical_levels.items()}
    numeric_stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "p1": float(df[col].quantile(0.01)),
            "p99": float(df[col].quantile(0.99)),
        }
        for col in numeric_features
    }
    feature_columns = [c for c in df.columns if c != target]

    feature_descriptions = {
        "sttl": "Source-to-destination total packets. Large counts in short sessions often come from scans or floods rather than normal user activity. Attack tools tend to generate bursts of packets to probe or overwhelm targets.",
        "dload": "Bytes per second from destination. Sudden download spikes can indicate data exfiltration or bulk error responses from a stressed service. Normal flows are steadier and tied to expected file sizes.",
        "sload": "Bytes per second from source. Sustained high upload rates can reflect flooding, beaconing, or brute-force attempts. Benign clients usually send at rates matching application needs.",
        "rate": "Packets per second. Malicious traffic is often bursty or unnaturally consistent compared to human-driven sessions. High packet rates without corresponding session context are suspicious.",
        "sbytes": "Total bytes from source. Very large sends in short-lived connections can signal exfiltration or DoS payloads. Normal interactions usually align with content requested by the server.",
        "smean": "Mean packet size from source. Attack scripts may emit uniform packet sizes; benign traffic tends to vary based on payload type. Uniformity or extremes can hint at automated attacks.",
        "dmean": "Mean packet size from destination. Oddly small or uniform responses can be signs of scanning, errors, or probing attempts rather than regular application replies.",
        "dttl": "Destination time-to-live. Unusual TTL values can indicate crafted packets or attempts to evade detection. Typical traffic stays within expected TTL ranges for the network path.",
        "dbytes": "Total bytes from destination. Unexpectedly large responses (or none) in a short window can reveal data dumps, errors, or failed handshake behavior.",
        "sloss": "Source packet loss count. High loss from the source side can indicate congestion from flooding or deliberate packet dropping by defensive devices. Benign flows usually see minimal loss in stable networks.",
        "tcprtt": "TCP round-trip time. Abnormally fast or slow RTTs, relative to other features, can indicate spoofing, congestion from floods, or distant command-and-control. Benign flows usually show RTTs consistent with path latency.",
        "synack": "Time between SYN and ACK. Long SYN-ACK times or anomalies suggest handshake issues from scanning or DoS. Normal sessions complete the handshake quickly.",
        "ackdat": "Time between ACK and data. Delays or abnormal patterns can reflect slowloris-style attacks or automated tools. Healthy connections start sending data promptly after the handshake.",
        "dinpkt": "Destination inter-packet arrival time. Very short gaps can mean bursts from floods; long erratic gaps may indicate probing or retries. Normal flows have more predictable pacing.",
        "sinpkt": "Source inter-packet arrival time. Burstiness or unnaturally regular timing often comes from automated attack scripts rather than human-driven traffic.",
        "dur": "Connection duration. Very short noisy sessions point to scans; very long sessions with steady traffic can indicate tunnels or exfiltration. Normal durations match user actions.",
        "proto": "Protocol type (tcp/udp/icmp…). Certain attacks favor specific protocols or unusual mixes (e.g., unexpected ICMP). Seeing rare protocols in a given environment can raise suspicion.",
        "state": "Connection state (e.g., CON, FIN, REJ). Many rejected/half-open states suggest scanning or DoS. Benign traffic tends to complete handshakes and close cleanly.",
    }

    # Build per-class example rows (up to 10 real rows sampled) for UI presets
    class_examples: dict[str, list[dict]] = {}
    for cls in sorted(df[target].unique()):
        subset = df[df[target] == cls]
        sample = subset.sample(n=min(10, len(subset)), random_state=RANDOM_STATE)
        rows = []
        for _, row in sample.iterrows():
            rows.append({col: row[col] for col in feature_columns})
        class_examples[str(cls)] = rows

    # Human-friendly labels (fallback to raw class)
    class_labels = {}
    unique_classes = sorted(df[target].unique())
    for cls in unique_classes:
        if str(cls) == "0":
            class_labels[str(cls)] = "benign (normal)"
        elif str(cls) == "1":
            class_labels[str(cls)] = "attack"
        else:
            class_labels[str(cls)] = str(cls)

    metadata = {
        "target": target,
        "feature_columns": feature_columns,
        "categorical_levels": categorical_levels,
        "categorical_defaults": categorical_defaults,
        "numeric_stats": numeric_stats,
        "top_features": top_features,
        "top_features_by_model": {},
        "feature_descriptions": feature_descriptions,
        "class_examples": class_examples,
        "class_labels": class_labels,
        "drop_always": DROP_ALWAYS,
        "wireshark_non_scannable": WIRESHARK_NON_SCANNABLE,
    }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest on UNSW-NB15 and save artifacts.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to UNSW_NB15_training-set.csv")
    parser.add_argument("--target", type=str, default="label", choices=["label", "attack_cat"], help="Target column")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Where to save artifacts")
    parser.add_argument("--live-features", action="store_true", help="Restrict features to those available from live capture (reduces mismatches).")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in dataset columns.")

    df = df.drop(columns=DROP_ALWAYS + WIRESHARK_NON_SCANNABLE, errors="ignore")
    y = df[args.target]
    X = df.drop(columns=[args.target, "attack_cat"], errors="ignore")

    if args.live_features:
        live_cols = [
            "sttl",
            "dttl",
            "sbytes",
            "dbytes",
            "sload",
            "dload",
            "rate",
            "smean",
            "dmean",
            "dur",
            "proto",
        ]
        existing = [c for c in live_cols if c in X.columns]
        X = X[existing]
        df = pd.concat([X, y], axis=1)

    preprocessor, categorical_features, numeric_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, n_jobs=-1, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic" if len(y.unique()) == 2 else "multi:softprob",
        )

    results = []
    fitted = {}
    top_by_model: dict[str, list[str]] = {}
    for name, est in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", est),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        results.append(
            {
                "model": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        fitted[name] = pipe
        print(f"\n{name} classification report (test):")
        print(classification_report(y_test, y_pred, zero_division=0))

        preprocess = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        feature_names = preprocess.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).ravel()
        else:
            importances = np.zeros(len(feature_names))
        aggregated = aggregate_importances(feature_names, importances)
        top_by_model[name] = [feat for feat, _ in aggregated[:15]]

    # Best model selection
    best = max(results, key=lambda r: r["f1"])
    best_name = best["model"]
    best_pipe = fitted[best_name]

    # Build union of top features, keeping best model's order first
    union_top = list(dict.fromkeys(top_by_model.get(best_name, [])))
    for feats in top_by_model.values():
        for f in feats:
            if f not in union_top:
                union_top.append(f)
    top_features = union_top[:20]

    metadata = collect_metadata(df, categorical_features, numeric_features, args.target, top_features)
    metadata["best_model"] = best_name
    metadata["model_results"] = results
    metadata["top_features_by_model"] = top_by_model
    metadata["live_features"] = bool(args.live_features)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, args.artifacts_dir / "models.joblib")
    with open(args.artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest by weighted F1: {best_name} (f1={best['f1']:.3f})")
    print(f"Saved models to {args.artifacts_dir / 'models.joblib'}")
    print(f"Saved metadata to {args.artifacts_dir / 'metadata.json'}")
    print(f"Top features: {', '.join(top_features)}")


if __name__ == "__main__":
    main()
