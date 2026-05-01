#!/usr/bin/env python3
# ============================================================
# UGR16 — Multi-Model LLM Binary Evaluation Script
# Enhanced Version 2.0
#
# Purpose:
#   Evaluate multiple pretrained LLM APIs on UGR16 NetFlow-style
#   records using a reproducible binary classification task:
#       attack / normal
#
# Main improvements over v1:
#   1. No hard-coded API keys.
#   2. API keys are read from environment variables.
#   3. Resume mode: already processed records are skipped.
#   4. Retry + exponential backoff for temporary API/network errors.
#   5. Safer prediction parsing: exact/structured parsing instead of
#      simple substring matching.
#   6. Per-model prediction files are written incrementally.
#   7. Failed calls are stored and reported.
#   8. Better metrics: precision, recall, F1, false positive rate,
#      false negative rate, specificity, sensitivity, accuracy.
#   9. Research metadata saved for reproducibility.
#
# Required input files:
#   sample_1000_api.jsonl   -> records WITHOUT label
#   sample_1000_eval.jsonl  -> records WITH label
#
# Recommended run:
#   export OPENAI_API_KEY="..."
#   export GEMINI_API_KEY="..."
#   export DEEPSEEK_API_KEY="..."
#   export ANTHROPIC_API_KEY="..."
#
#   python3 ugr16_llm_eval_v2.py \
#       --sample-dir /data/llm_samples_v10 \
#       --output-dir /data/llm_eval_results_v2 \
#       --sample-size 1000 \
#       --models chatgpt gemini deepseek claude
#
# ============================================================

import argparse
import datetime as dt
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests


# ============================================================
# Default model configuration
# Update model names only after checking the official API docs.
# ============================================================

DEFAULT_MODEL_CONFIG = {
    "claude": {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "chatgpt": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider": "google",
        "model": "gemini-1.5-flash-latest",
        "env_key": "GEMINI_API_KEY",
    },
    "deepseek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
}


# ============================================================
# Approximate cost estimates per 1000 tokens, USD.
# Important:
#   These are placeholders for monitoring only.
#   Update from official pricing pages before final thesis use.
# ============================================================

COST_PER_1K_TOKENS = {
    "claude": {"input": 0.003, "output": 0.015},
    "chatgpt": {"input": 0.00015, "output": 0.00060},
    "gemini": {"input": 0.00025, "output": 0.00050},
    "deepseek": {"input": 0.00014, "output": 0.00028},
}

APPROX_INPUT_TOKENS_PER_RECORD = 300
APPROX_OUTPUT_TOKENS_PER_RECORD = 10


# ============================================================
# Prompt
# ============================================================

def build_prompt(record: Dict[str, Any]) -> str:
    """
    Build a consistent prompt for all models.

    The model is forced to return a very small JSON object.
    This is safer than asking for free text, and easier to parse.
    """
    record_json = json.dumps(record, ensure_ascii=False, sort_keys=True)

    return f"""You are a network security analyst specialising in intrusion detection.

Classify the following network flow record as either attack or normal.

Network flow JSON:
{record_json}

Field meanings:
- timestamp: capture time
- duration: flow duration in seconds
- src_ip / dst_ip: source and destination IP addresses
- src_port / dst_port: source and destination ports
- protocol: network protocol
- flags: TCP flags
- fwd_pkts / bwd_pkts: forward and backward packet counts
- total_pkts: total packets in the flow
- total_bytes: total bytes transferred

Rules:
- Use only the provided flow features.
- Do not invent missing information.
- If uncertain, choose normal.
- Return valid JSON only.
- The label must be exactly one of: attack, normal.

Required JSON format:
{{"label":"attack","confidence":0.0,"reason":"short reason"}}
"""


# ============================================================
# Label mapping
# ============================================================

def map_label_to_binary(label: str) -> str:
    """
    Convert UGR16 labels into binary labels.

    Current mapping:
      background      -> normal
      blacklist       -> attack
      anomaly-sshscan -> attack
      any other label -> attack

    This should be explained in the thesis methodology.
    """
    label_clean = str(label).strip().lower()
    if label_clean == "background":
        return "normal"
    return "attack"


# ============================================================
# Utility functions
# ============================================================

def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return records


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def prediction_file(output_dir: Path, sample_size: int, model_name: str) -> Path:
    return output_dir / f"predictions_{sample_size}_{model_name}.jsonl"


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read an existing prediction file to support resume mode.
    Returns a lookup by record_id.
    """
    if not path.exists():
        return {}

    existing = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rid = str(row.get("record_id", ""))
                if rid:
                    existing[rid] = row
            except json.JSONDecodeError:
                continue
    return existing


def remove_sensitive_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prevent label leakage into the LLM prompt.

    The API file should already exclude these fields, but this function
    makes the script safer if the wrong file is accidentally used.
    """
    blocked = {
        "label",
        "ground_truth",
        "true_label",
        "binary_label",
        "suspect_reason",
        "is_attack",
        "attack_type",
    }
    return {k: v for k, v in record.items() if k not in blocked}


def parse_model_response(text: str) -> Tuple[str, Optional[float], str]:
    """
    Safely parse model output.

    Preferred output is JSON:
      {"label":"attack","confidence":0.8,"reason":"..."}

    Fallback:
      Accept exact single word: attack / normal

    We do NOT use substring logic like:
      "attack" in text
    because "not attack" would be incorrectly classified as attack.
    """
    raw = (text or "").strip()

    # Try JSON parsing first.
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip().lower()
        confidence = data.get("confidence", None)
        reason = str(data.get("reason", "")).strip()

        if label not in {"attack", "normal"}:
            raise ValueError(f"Invalid label: {label}")

        if confidence is not None:
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = None

        return label, confidence, reason

    except Exception:
        pass

    # Try to extract JSON block if the model wrapped it in text.
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            label = str(data.get("label", "")).strip().lower()
            confidence = data.get("confidence", None)
            reason = str(data.get("reason", "")).strip()

            if label in {"attack", "normal"}:
                if confidence is not None:
                    try:
                        confidence = float(confidence)
                        confidence = max(0.0, min(1.0, confidence))
                    except (TypeError, ValueError):
                        confidence = None
                return label, confidence, reason
        except Exception:
            pass

    # Exact one-word fallback only.
    cleaned = raw.lower().strip().strip(".:;!\"'")
    if cleaned in {"attack", "normal"}:
        return cleaned, None, "single-word fallback"

    raise ValueError(f"Could not parse valid prediction from response: {raw[:200]}")


def sleep_with_backoff(attempt: int, base_sleep: float = 2.0) -> None:
    """
    Exponential backoff with small jitter.
    attempt starts from 1.
    """
    delay = base_sleep * (2 ** (attempt - 1))
    delay += random.uniform(0, 0.5)
    time.sleep(delay)


# ============================================================
# API callers
# ============================================================

def post_with_retries(
    url: str,
    headers: Optional[Dict[str, str]],
    body: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> requests.Response:
    """
    Retry temporary failures:
      - 429 rate limit
      - 500/502/503/504 server errors
      - timeout / connection errors
    """
    retry_statuses = {429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)

            if response.status_code in retry_statuses:
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
                sleep_with_backoff(attempt)
                continue

            response.raise_for_status()
            return response

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_error = exc
            if attempt < max_retries:
                sleep_with_backoff(attempt)
                continue
            raise

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


def call_claude(
    prompt: str,
    api_key: str,
    model_id: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 80,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }

    t0 = time.time()
    r = post_with_retries(url, headers, body, timeout, max_retries)
    latency = round(time.time() - t0, 3)
    data = r.json()

    text = data["content"][0]["text"]
    label, confidence, reason = parse_model_response(text)

    usage = data.get("usage", {})
    return {
        "raw_response": text,
        "prediction": label,
        "confidence": confidence,
        "reason": reason,
        "input_tokens": int(usage.get("input_tokens", APPROX_INPUT_TOKENS_PER_RECORD)),
        "output_tokens": int(usage.get("output_tokens", APPROX_OUTPUT_TOKENS_PER_RECORD)),
        "latency_s": latency,
    }


def call_chatgpt(
    prompt: str,
    api_key: str,
    model_id: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 80,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    r = post_with_retries(url, headers, body, timeout, max_retries)
    latency = round(time.time() - t0, 3)
    data = r.json()

    text = data["choices"][0]["message"]["content"]
    label, confidence, reason = parse_model_response(text)

    usage = data.get("usage", {})
    return {
        "raw_response": text,
        "prediction": label,
        "confidence": confidence,
        "reason": reason,
        "input_tokens": int(usage.get("prompt_tokens", APPROX_INPUT_TOKENS_PER_RECORD)),
        "output_tokens": int(usage.get("completion_tokens", APPROX_OUTPUT_TOKENS_PER_RECORD)),
        "latency_s": latency,
    }


def call_gemini(
    prompt: str,
    api_key: str,
    model_id: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 80,
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    t0 = time.time()
    r = post_with_retries(url, None, body, timeout, max_retries)
    latency = round(time.time() - t0, 3)
    data = r.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    label, confidence, reason = parse_model_response(text)

    usage = data.get("usageMetadata", {})
    return {
        "raw_response": text,
        "prediction": label,
        "confidence": confidence,
        "reason": reason,
        "input_tokens": int(usage.get("promptTokenCount", APPROX_INPUT_TOKENS_PER_RECORD)),
        "output_tokens": int(usage.get("candidatesTokenCount", APPROX_OUTPUT_TOKENS_PER_RECORD)),
        "latency_s": latency,
    }


def call_deepseek(
    prompt: str,
    api_key: str,
    model_id: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 80,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    r = post_with_retries(url, headers, body, timeout, max_retries)
    latency = round(time.time() - t0, 3)
    data = r.json()

    text = data["choices"][0]["message"]["content"]
    label, confidence, reason = parse_model_response(text)

    usage = data.get("usage", {})
    return {
        "raw_response": text,
        "prediction": label,
        "confidence": confidence,
        "reason": reason,
        "input_tokens": int(usage.get("prompt_tokens", APPROX_INPUT_TOKENS_PER_RECORD)),
        "output_tokens": int(usage.get("completion_tokens", APPROX_OUTPUT_TOKENS_PER_RECORD)),
        "latency_s": latency,
    }


CALLERS = {
    "claude": call_claude,
    "chatgpt": call_chatgpt,
    "gemini": call_gemini,
    "deepseek": call_deepseek,
}


# ============================================================
# Metrics
# ============================================================

def compute_metrics(predictions: Iterable[Dict[str, Any]], eval_lookup: Dict[str, str]) -> Dict[str, Any]:
    TP = FP = TN = FN = missing_truth = invalid_prediction = 0

    for p in predictions:
        rid = str(p.get("record_id", ""))
        pred = str(p.get("prediction", "")).strip().lower()
        true = eval_lookup.get(rid)

        if true is None:
            missing_truth += 1
            continue

        if pred not in {"attack", "normal"}:
            invalid_prediction += 1
            continue

        if true == "attack" and pred == "attack":
            TP += 1
        elif true == "normal" and pred == "attack":
            FP += 1
        elif true == "normal" and pred == "normal":
            TN += 1
        elif true == "attack" and pred == "normal":
            FN += 1

    total_valid = TP + FP + TN + FN

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    sensitivity = recall
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    fnr = FN / (FN + TP) if (FN + TP) else 0.0
    accuracy = (TP + TN) / total_valid if total_valid else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "total_valid": total_valid,
        "missing_truth": missing_truth,
        "invalid_prediction": invalid_prediction,
        "precision": round(precision, 6),
        "recall_sensitivity": round(sensitivity, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "accuracy": round(accuracy, 6),
    }


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    prices = COST_PER_1K_TOKENS.get(model_name, {"input": 0, "output": 0})
    return round(
        ((input_tokens / 1000) * prices["input"])
        + ((output_tokens / 1000) * prices["output"]),
        6,
    )


# ============================================================
# Reporting
# ============================================================

def print_checkpoint(
    completed_records: int,
    total_records: int,
    started_at: float,
    model_stats: Dict[str, Dict[str, Any]],
) -> None:
    elapsed = time.time() - started_at
    pct = (completed_records / total_records) * 100 if total_records else 0
    done = int((pct / 100) * 30)
    bar = "█" * done + "░" * (30 - done)

    eta = 0
    if completed_records > 0:
        eta = (elapsed / completed_records) * (total_records - completed_records)

    print("\n" + "=" * 80)
    print(f"CHECKPOINT: {completed_records}/{total_records} [{bar}] {pct:.1f}%")
    print(f"Elapsed   : {dt.timedelta(seconds=int(elapsed))}")
    print(f"ETA       : {dt.timedelta(seconds=int(eta))}")
    print("-" * 80)
    print(
        f"{'Model':<12} {'Calls':>7} {'Errors':>7} {'InTok':>12} "
        f"{'OutTok':>10} {'Cost($)':>10} {'AvgLat(s)':>10}"
    )
    print("-" * 80)

    total_cost = 0.0
    for model_name, s in model_stats.items():
        calls = s["calls"]
        errors = s["errors"]
        in_tok = s["input_tokens"]
        out_tok = s["output_tokens"]
        cost = estimate_cost(model_name, in_tok, out_tok)
        total_cost += cost
        avg_lat = s["total_latency"] / calls if calls else 0.0
        print(
            f"{model_name:<12} {calls:>7} {errors:>7} {in_tok:>12,} "
            f"{out_tok:>10,} {cost:>10.6f} {avg_lat:>10.3f}"
        )

    print("-" * 80)
    print(f"Current estimated total cost: ${total_cost:.6f}")
    print("=" * 80)


def write_text_report(
    path: Path,
    args: argparse.Namespace,
    attack_count: int,
    normal_count: int,
    model_stats: Dict[str, Dict[str, Any]],
    all_metrics: Dict[str, Dict[str, Any]],
    total_elapsed: float,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        W = f.write
        W("UGR16 Multi-Model LLM Evaluation Report — Enhanced v2\n")
        W("=" * 70 + "\n\n")
        W(f"Generated at       : {now_iso()}\n")
        W(f"Task               : Binary classification attack/normal\n")
        W(f"Sample size        : {args.sample_size}\n")
        W(f"API input file     : {args.api_file}\n")
        W(f"Eval file          : {args.eval_file}\n")
        W(f"Models             : {', '.join(args.models)}\n")
        W(f"Total runtime      : {dt.timedelta(seconds=int(total_elapsed))}\n")
        W(f"Ground truth attack: {attack_count}\n")
        W(f"Ground truth normal: {normal_count}\n\n")

        W("Method summary\n")
        W("-" * 70 + "\n")
        W(
            "Each JSONL record was sent independently to the selected pretrained LLM APIs. "
            "The prompt asked for a binary classification using only the flow features. "
            "The model response was required to be JSON with label, confidence, and reason. "
            "Predictions were joined with the evaluation JSONL file using record_id.\n\n"
        )

        W("Metrics\n")
        W("-" * 70 + "\n")
        for model_name, m in all_metrics.items():
            W(f"\n{model_name.upper()}\n")
            W(f"  TP                  : {m['TP']}\n")
            W(f"  FP                  : {m['FP']}\n")
            W(f"  TN                  : {m['TN']}\n")
            W(f"  FN                  : {m['FN']}\n")
            W(f"  Precision           : {m['precision']}\n")
            W(f"  Recall/Sensitivity  : {m['recall_sensitivity']}\n")
            W(f"  Specificity         : {m['specificity']}\n")
            W(f"  F1-score            : {m['f1']}\n")
            W(f"  False Positive Rate : {m['false_positive_rate']}\n")
            W(f"  False Negative Rate : {m['false_negative_rate']}\n")
            W(f"  Accuracy            : {m['accuracy']}\n")
            W(f"  Valid predictions   : {m['total_valid']}\n")
            W(f"  Missing truth       : {m['missing_truth']}\n")
            W(f"  Invalid predictions : {m['invalid_prediction']}\n")

        W("\nCost and runtime summary\n")
        W("-" * 70 + "\n")
        for model_name, s in model_stats.items():
            cost = estimate_cost(model_name, s["input_tokens"], s["output_tokens"])
            avg_lat = s["total_latency"] / s["calls"] if s["calls"] else 0.0
            W(f"\n{model_name.upper()}\n")
            W(f"  Calls        : {s['calls']}\n")
            W(f"  Errors       : {s['errors']}\n")
            W(f"  Input tokens : {s['input_tokens']}\n")
            W(f"  Output tokens: {s['output_tokens']}\n")
            W(f"  Avg latency  : {avg_lat:.3f} seconds\n")
            W(f"  Est. cost    : ${cost:.6f}\n")


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UGR16 multi-model LLM binary evaluation script v2."
    )

    parser.add_argument(
        "--sample-dir",
        default="/data/llm_samples_v10",
        help="Directory containing sample API/eval JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/llm_eval_results_v2",
        help="Directory to save predictions and reports.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size used in file names, e.g. 1000, 3000, 5000.",
    )
    parser.add_argument(
        "--api-file",
        default=None,
        help="Optional explicit API JSONL path. Overrides --sample-dir/--sample-size.",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Optional explicit eval JSONL path. Overrides --sample-dir/--sample-size.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["chatgpt", "gemini", "deepseek", "claude"],
        choices=list(DEFAULT_MODEL_CONFIG.keys()),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Print progress every N records.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per API call.",
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Optional sleep between API calls to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and overwrite old prediction files.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional limit for testing, e.g. --limit-records 10.",
    )

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    args.output_dir = Path(args.output_dir)
    args.api_file = Path(args.api_file) if args.api_file else sample_dir / f"sample_{args.sample_size}_api.jsonl"
    args.eval_file = Path(args.eval_file) if args.eval_file else sample_dir / f"sample_{args.sample_size}_eval.jsonl"

    return args


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("UGR16 LLM Evaluation — Enhanced v2")
    print(f"Started       : {now_iso()}")
    print(f"Task          : binary classification attack/normal")
    print(f"Models        : {', '.join(args.models)}")
    print(f"API file      : {args.api_file}")
    print(f"Eval file     : {args.eval_file}")
    print(f"Output dir    : {args.output_dir}")
    print("=" * 80)

    if not args.api_file.exists():
        raise FileNotFoundError(f"API file not found: {args.api_file}")
    if not args.eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    # Validate API keys before starting.
    api_keys = {}
    missing_keys = []
    for model_name in args.models:
        env_key = DEFAULT_MODEL_CONFIG[model_name]["env_key"]
        api_key = os.getenv(env_key, "").strip()
        if not api_key:
            missing_keys.append(f"{model_name}: missing {env_key}")
        api_keys[model_name] = api_key

    if missing_keys:
        print("\nMissing API keys:")
        for item in missing_keys:
            print(f"  - {item}")
        raise SystemExit(
            "\nSet the required environment variables before running. "
            "Do not write API keys inside the script."
        )

    # Load records.
    api_records = read_jsonl(args.api_file)
    if args.limit_records is not None:
        api_records = api_records[: args.limit_records]

    total_records = len(api_records)
    print(f"\nLoaded API records: {total_records:,}")

    # Load ground truth.
    eval_records = read_jsonl(args.eval_file)
    eval_lookup = {}
    for rec in eval_records:
        rid = str(rec.get("record_id", ""))
        if not rid:
            continue
        eval_lookup[rid] = map_label_to_binary(str(rec.get("label", "")))

    attack_count = sum(1 for v in eval_lookup.values() if v == "attack")
    normal_count = sum(1 for v in eval_lookup.values() if v == "normal")
    print(f"Ground truth: {attack_count:,} attack, {normal_count:,} normal")

    # Prepare prediction files and resume state.
    existing_by_model = {}
    if args.no_resume:
        for model_name in args.models:
            pfile = prediction_file(args.output_dir, args.sample_size, model_name)
            if pfile.exists():
                pfile.unlink()

    for model_name in args.models:
        pfile = prediction_file(args.output_dir, args.sample_size, model_name)
        existing_by_model[model_name] = load_existing_predictions(pfile)

    model_stats = {
        model_name: {
            "calls": 0,
            "errors": 0,
            "skipped_resume": len(existing_by_model[model_name]),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_latency": 0.0,
            "model_id": DEFAULT_MODEL_CONFIG[model_name]["model"],
            "env_key": DEFAULT_MODEL_CONFIG[model_name]["env_key"],
        }
        for model_name in args.models
    }

    errors_path = args.output_dir / f"errors_{args.sample_size}.jsonl"
    started_at = time.time()

    # Main loop.
    for idx, original_record in enumerate(api_records, start=1):
        record_id = str(original_record.get("record_id", ""))

        if not record_id:
            append_jsonl(errors_path, {
                "timestamp": now_iso(),
                "record_index": idx,
                "record_id": None,
                "error": "missing record_id",
            })
            continue

        safe_record = remove_sensitive_fields(original_record)
        prompt = build_prompt(safe_record)

        for model_name in args.models:
            if record_id in existing_by_model[model_name]:
                continue

            config = DEFAULT_MODEL_CONFIG[model_name]
            model_id = config["model"]
            caller = CALLERS[model_name]

            try:
                result = caller(
                    prompt=prompt,
                    api_key=api_keys[model_name],
                    model_id=model_id,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )

                row = {
                    "timestamp": now_iso(),
                    "record_index": idx,
                    "record_id": record_id,
                    "model_name": model_name,
                    "model_id": model_id,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "reason": result["reason"],
                    "raw_response": result["raw_response"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_s": result["latency_s"],
                    "status": "ok",
                }

                append_jsonl(prediction_file(args.output_dir, args.sample_size, model_name), row)
                existing_by_model[model_name][record_id] = row

                model_stats[model_name]["calls"] += 1
                model_stats[model_name]["input_tokens"] += result["input_tokens"]
                model_stats[model_name]["output_tokens"] += result["output_tokens"]
                model_stats[model_name]["total_latency"] += result["latency_s"]

            except Exception as exc:
                model_stats[model_name]["errors"] += 1

                error_row = {
                    "timestamp": now_iso(),
                    "record_index": idx,
                    "record_id": record_id,
                    "model_name": model_name,
                    "model_id": model_id,
                    "error": str(exc),
                    "status": "error",
                }

                append_jsonl(errors_path, error_row)

                # Also write an error prediction row, so missing records are visible.
                append_jsonl(prediction_file(args.output_dir, args.sample_size, model_name), {
                    **error_row,
                    "prediction": "error",
                    "confidence": None,
                    "reason": "api_or_parse_error",
                    "raw_response": str(exc),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0,
                })

            if args.sleep_between_calls > 0:
                time.sleep(args.sleep_between_calls)

        if idx % args.checkpoint_every == 0 or idx == total_records:
            print_checkpoint(idx, total_records, started_at, model_stats)

    # Final metrics.
    all_metrics = {}
    for model_name in args.models:
        preds = list(load_existing_predictions(
            prediction_file(args.output_dir, args.sample_size, model_name)
        ).values())
        all_metrics[model_name] = compute_metrics(preds, eval_lookup)

    total_elapsed = time.time() - started_at

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(
        f"{'Model':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'FPR':>10} {'FNR':>10} {'Accuracy':>10} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}"
    )
    print("-" * 100)

    for model_name, m in all_metrics.items():
        print(
            f"{model_name:<12} "
            f"{m['precision']:>10.4f} "
            f"{m['recall_sensitivity']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{m['false_positive_rate']:>10.4f} "
            f"{m['false_negative_rate']:>10.4f} "
            f"{m['accuracy']:>10.4f} "
            f"{m['TP']:>6} "
            f"{m['FP']:>6} "
            f"{m['TN']:>6} "
            f"{m['FN']:>6}"
        )

    print("=" * 100)
    print(f"Total runtime: {dt.timedelta(seconds=int(total_elapsed))}")

    # Save JSON report.
    report = {
        "script": "ugr16_llm_eval_v2.py",
        "version": "2.0",
        "generated_at": now_iso(),
        "task": "binary_classification_attack_normal",
        "method": {
            "description": (
                "Independent per-record LLM classification using pretrained APIs. "
                "Predictions are joined with ground truth by record_id."
            ),
            "label_mapping": {
                "background": "normal",
                "blacklist": "attack",
                "anomaly-sshscan": "attack",
                "other": "attack",
            },
            "prompt_style": "JSON output with label, confidence, and reason",
            "resume_enabled": not args.no_resume,
        },
        "input_files": {
            "api_file": str(args.api_file),
            "eval_file": str(args.eval_file),
        },
        "sample_size": args.sample_size,
        "limit_records": args.limit_records,
        "models": {
            model_name: {
                "model_id": DEFAULT_MODEL_CONFIG[model_name]["model"],
                "provider": DEFAULT_MODEL_CONFIG[model_name]["provider"],
            }
            for model_name in args.models
        },
        "ground_truth": {
            "attack": attack_count,
            "normal": normal_count,
        },
        "runtime": {
            "total_seconds": round(total_elapsed, 3),
            "total_hms": str(dt.timedelta(seconds=int(total_elapsed))),
        },
        "model_stats": model_stats,
        "metrics": all_metrics,
        "cost_note": "Cost values are estimates only. Update prices from official provider pages before thesis submission.",
    }

    json_report_path = args.output_dir / f"eval_report_{args.sample_size}.json"
    with json_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt_report_path = args.output_dir / f"eval_report_{args.sample_size}.txt"
    write_text_report(
        txt_report_path,
        args,
        attack_count,
        normal_count,
        model_stats,
        all_metrics,
        total_elapsed,
    )

    print("\nReports saved:")
    print(f"  JSON: {json_report_path}")
    print(f"  TXT : {txt_report_path}")
    print(f"  Errors, if any: {errors_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
