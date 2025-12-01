"""
Live network monitor for UNSW-NB15 model.

Captures packets (macOS default interface auto-detected), aggregates simple flow
features, feeds them through the saved preprocessing + model pipeline, and
prints anomalies (predicted label != benign or attack probability over
threshold). Remains silent when nothing suspicious is seen.

Requirements:
  - Python deps: pip install -r requirements.txt
  - Run with privileges that allow packet capture (e.g., sudo on macOS).

Usage:
  python live_monitor.py --data-path UNSW-NB15/UNSW_NB15_training-set.csv
    (not needed if artifacts already exist)
  python live_monitor.py --iface auto --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, get_working_ifaces, Packet  # type: ignore


ARTIFACTS_DIR = Path("artifacts")
MODELS_PATH = ARTIFACTS_DIR / "models.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEFAULT_DATA_PATH = Path("UNSW-NB15/UNSW_NB15_training-set.csv")
DEFAULT_DATA_PATH = Path("UNSW-NB15/UNSW_NB15_training-set.csv")


def pick_iface() -> str:
    """Choose a reasonable default interface on macOS (prefer en0)."""
    try:
        ifaces = [iface.name for iface in get_working_ifaces() if iface.is_up()]
    except Exception:
        return "en0"
    for preferred in ("en0", "en1", "eth0", "eth1", "wlan0"):
        if preferred in ifaces:
            return preferred
    return ifaces[0] if ifaces else "en0"


@dataclass
class FlowStats:
    src: str
    dst: str
    sport: int
    dport: int
    proto: str
    first_ts: float
    last_ts: float
    pkts: int = 0
    bytes: int = 0
    sbytes: int = 0
    dbytes: int = 0
    sttl: int | None = None
    dttl: int | None = None
    sizes_src: list = field(default_factory=list)
    sizes_dst: list = field(default_factory=list)
    last_alert: Dict[str, float] = field(default_factory=dict)  # model name -> timestamp

    def update(self, pkt: Packet, ts: float, direction_src: bool, length: int, ttl: int | None):
        self.pkts += 1
        self.bytes += length
        self.last_ts = ts
        if direction_src:
            self.sbytes += length
            self.sizes_src.append(length)
            if ttl is not None:
                self.sttl = ttl
        else:
            self.dbytes += length
            self.sizes_dst.append(length)
            if ttl is not None:
                self.dttl = ttl

    def to_features(self) -> Dict[str, Any]:
        dur = max(self.last_ts - self.first_ts, 1e-6)
        rate = self.pkts / dur
        sload = self.sbytes / dur
        dload = self.dbytes / dur
        smean = np.mean(self.sizes_src) if self.sizes_src else 0.0
        dmean = np.mean(self.sizes_dst) if self.sizes_dst else 0.0
        return {
            "sttl": float(self.sttl) if self.sttl is not None else np.nan,
            "dttl": float(self.dttl) if self.dttl is not None else np.nan,
            "sbytes": float(self.sbytes),
            "dbytes": float(self.dbytes),
            "sload": float(sload),
            "dload": float(dload),
            "rate": float(rate),
            "smean": float(smean),
            "dmean": float(dmean),
            "dur": float(dur),
            "proto": self.proto,
            # Placeholders for model columns we cannot derive; filled via defaults later.
        }


class FlowTable:
    def __init__(self, max_flows: int = 5000, idle_timeout: float = 60.0):
        self.max_flows = max_flows
        self.idle_timeout = idle_timeout
        self.flows: Dict[Tuple[str, str, int, int, str], FlowStats] = {}

    def update(self, pkt: Packet, ts: float):
        # Extract IP/port/proto
        if not pkt.haslayer("IP"):
            return None
        ip = pkt.getlayer("IP")
        proto_num = ip.proto
        proto = {6: "tcp", 17: "udp", 1: "icmp"}.get(proto_num, str(proto_num))
        src = ip.src
        dst = ip.dst
        ttl = int(ip.ttl) if hasattr(ip, "ttl") else None
        sport = int(getattr(pkt, "sport", 0) or 0)
        dport = int(getattr(pkt, "dport", 0) or 0)
        length = int(getattr(pkt, "len", len(bytes(pkt))) or 0)

        key = (src, dst, sport, dport, proto)
        reverse_key = (dst, src, dport, sport, proto)

        direction_src = True
        flow = self.flows.get(key)
        if flow is None and reverse_key in self.flows:
            flow = self.flows[reverse_key]
            direction_src = False
        if flow is None:
            if len(self.flows) >= self.max_flows:
                # drop oldest
                oldest = min(self.flows.values(), key=lambda f: f.last_ts)
                self.flows = {k: v for k, v in self.flows.items() if v is not oldest}
            flow = FlowStats(src=src, dst=dst, sport=sport, dport=dport, proto=proto, first_ts=ts, last_ts=ts)
            self.flows[key] = flow

        flow.update(pkt, ts, direction_src, length, ttl)
        return flow

    def harvest_idle(self, now: float):
        to_remove = [k for k, f in self.flows.items() if (now - f.last_ts) > self.idle_timeout]
        for k in to_remove:
            self.flows.pop(k, None)

    def snapshot(self):
        return list(self.flows.values())


def ensure_artifacts():
    if MODELS_PATH.exists() and METADATA_PATH.exists():
        return
    data_path = Path(os.environ.get("DATA_PATH", DEFAULT_DATA_PATH))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Artifacts missing and dataset not found at {data_path}. "
            "Provide DATA_PATH env var or run train_model.py manually."
        )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "train_model.py",
        "--data-path",
        str(data_path),
        "--target",
        "label",
        "--live-features",
    ]
    subprocess.run(cmd, check=True)


def load_artifacts():
    ensure_artifacts()
    models = joblib.load(MODELS_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return models, metadata


def is_broadcast(ip: str) -> bool:
    return ip.endswith(".255")


def is_multicast(ip: str) -> bool:
    first_octet = int(ip.split(".")[0])
    return 224 <= first_octet <= 239


def fill_row(features: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    numeric_stats = metadata.get("numeric_stats", {})
    categorical_defaults = metadata.get("categorical_defaults", {})
    for col in metadata["feature_columns"]:
        if col in features and features[col] is not None:
            row[col] = features[col]
        elif col in numeric_stats:
            row[col] = numeric_stats[col].get("median", 0.0)
        elif col in categorical_defaults:
            row[col] = categorical_defaults[col]
        else:
            row[col] = 0.0
    return row


def predict(flows: list[FlowStats], models: Dict[str, Any], metadata: Dict[str, Any], threshold: float, now: float, min_pkts: int, min_bytes: int, cooldown: float, ignore_broadcast: bool, ignore_multicast: bool):
    if not flows:
        return []
    filtered_flows = []
    for f in flows:
        if f.pkts < min_pkts or f.bytes < min_bytes:
            continue
        if ignore_broadcast and (is_broadcast(f.dst) or is_broadcast(f.src)):
            continue
        if ignore_multicast and (is_multicast(f.dst) or is_multicast(f.src)):
            continue
        filtered_flows.append(f)
    if not filtered_flows:
        return []

    rows = [fill_row(f.to_features(), metadata) for f in filtered_flows]
    df = pd.DataFrame(rows, columns=metadata["feature_columns"])
    alerts = []
    class_labels = metadata.get("class_labels", {})
    benign_label = next((k for k, v in class_labels.items() if v.startswith("benign")), "0")

    for name, pipe in models.items():
        proba = pipe.predict_proba(df)
        preds = pipe.classes_[np.argmax(proba, axis=1)]
        for flow, p_vec, pred in zip(filtered_flows, proba, preds):
            pred_label = class_labels.get(str(pred), str(pred))
            attack_prob = float(max(p_vec)) if len(pipe.classes_) == 2 else float(1.0 - float(p_vec[pipe.classes_ == benign_label]) if np.any(pipe.classes_ == benign_label) else max(p_vec))
            is_anomaly = (str(pred) != str(benign_label)) or attack_prob >= threshold
            last = flow.last_alert.get(name, 0.0)
            if is_anomaly and (now - last) >= cooldown:
                alerts.append(
                    {
                        "model": name,
                        "flow": f"{flow.src}:{flow.sport} -> {flow.dst}:{flow.dport} ({flow.proto})",
                        "pred": pred_label,
                        "attack_prob": attack_prob,
                        "rate": flow.pkts / max(flow.last_ts - flow.first_ts, 1e-6),
                        "pkts": flow.pkts,
                        "bytes": flow.bytes,
                        "sbytes": flow.sbytes,
                        "dbytes": flow.dbytes,
                        "dur": max(flow.last_ts - flow.first_ts, 1e-6),
                    }
                )
                flow.last_alert[name] = now
    return alerts


def run_capture(args):
    models, metadata = load_artifacts()
    if not metadata.get("live_features", False):
        print("[warn] Artifacts were not trained with --live-features; expect more false positives. Retrain for live capture.", file=sys.stderr)
    iface = args.iface if args.iface != "auto" else pick_iface()
    flow_table = FlowTable(max_flows=args.max_flows, idle_timeout=args.idle_timeout)
    pkt_queue: queue.Queue = queue.Queue(maxsize=10000)

    stop_event = threading.Event()

    def handle_sig(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    def enqueue(pkt):
        try:
            pkt_queue.put_nowait((time.time(), pkt))
        except queue.Full:
            pass

    sniffer = threading.Thread(
        target=sniff,
        kwargs={"iface": iface, "prn": enqueue, "store": False},
        daemon=True,
    )
    sniffer.start()
    print(f"[monitor] capturing on {iface} (Ctrl+C to stop)", file=sys.stderr)

    last_check = time.time()
    while not stop_event.is_set():
        try:
            ts, pkt = pkt_queue.get(timeout=0.5)
        except queue.Empty:
            now = time.time()
            if now - last_check > args.idle_timeout:
                flow_table.harvest_idle(now)
                last_check = now
            continue

        flow_table.update(pkt, ts)

        if pkt_queue.qsize() >= args.batch_size:
            flows = flow_table.snapshot()
            alerts = predict(
                flows,
                models,
                metadata,
                threshold=args.threshold,
                now=time.time(),
                min_pkts=args.min_pkts,
                min_bytes=args.min_bytes,
                cooldown=args.cooldown,
                ignore_broadcast=args.ignore_broadcast,
                ignore_multicast=args.ignore_multicast,
            )
            for alert in alerts:
                print(
                    f"[{alert['model']}] {alert['flow']} -> {alert['pred']} "
                    f"(p={alert['attack_prob']:.2f}, rate={alert['rate']:.1f} pkts/s, "
                    f"pkts={alert['pkts']}, bytes={alert['bytes']}, sbytes={alert['sbytes']}, "
                    f"dbytes={alert['dbytes']}, dur={alert['dur']:.2f}s)"
                )
            flow_table.harvest_idle(time.time())


def main():
    parser = argparse.ArgumentParser(description="Live anomaly monitor using UNSW-NB15 model.")
    parser.add_argument("--iface", default="auto", help="Network interface to capture (default: auto-detect macOS en0/eth0).")
    parser.add_argument("--threshold", type=float, default=0.99, help="Attack probability threshold to alert.")
    parser.add_argument("--batch-size", type=int, default=50, help="Predict every N captured packets.")
    parser.add_argument("--min-pkts", type=int, default=10, help="Minimum packets in flow before scoring.")
    parser.add_argument("--min-bytes", type=int, default=200, help="Minimum bytes in flow before scoring.")
    parser.add_argument("--cooldown", type=float, default=60.0, help="Seconds before re-alerting on same flow/model.")
    parser.add_argument("--max-flows", type=int, default=5000, help="Max flows to track.")
    parser.add_argument("--idle-timeout", type=float, default=60.0, help="Seconds to keep idle flows.")
    parser.add_argument("--ignore-broadcast", action="store_true", help="Ignore broadcast addresses (x.x.x.255).")
    parser.add_argument("--ignore-multicast", action="store_true", help="Ignore multicast (224.0.0.0/4).")
    args = parser.parse_args()

    try:
        run_capture(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print("Permission denied: try running with sudo for packet capture.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
