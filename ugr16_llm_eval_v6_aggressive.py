#!/usr/bin/env python3
# ============================================================
# UGR16 — Multi-Model LLM Binary Evaluation Script
# Final Stable Version V6
# ============================================================

import argparse
import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_MODEL_CONFIG = {
    "chatgpt": {"provider": "openai", "model": "gpt-4.1-mini", "env_key": "OPENAI_API_KEY"},
    "gemini": {"provider": "google", "model": "gemini-2.5-flash", "env_key": "GEMINI_API_KEY"},
    "deepseek": {"provider": "deepseek", "model": "deepseek-chat", "env_key": "DEEPSEEK_API_KEY"},
    "claude": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "env_key": "ANTHROPIC_API_KEY"},
}

COST_PER_1K_TOKENS = {
    "claude": {"input": 0.0008, "output": 0.0040},
    "chatgpt": {"input": 0.00015, "output": 0.00060},
    "gemini": {"input": 0.000075, "output": 0.00030},
    "deepseek": {"input": 0.00014, "output": 0.00028},
}

APPROX_INPUT_TOKENS_PER_RECORD = 300
APPROX_OUTPUT_TOKENS_PER_RECORD = 10


def build_prompt(record: Dict[str, Any]) -> str:
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
- If there are any suspicious indicators, classify as attack.
- Do not default to normal when uncertain.
- Return JSON only. Do not use Markdown. Do not add explanations outside JSON.
- The label must be exactly one of: attack, normal.
- confidence must be a number from 0.0 to 1.0.
- reason must be short, maximum 20 words.

Return ONLY this compact JSON, no explanation:
{{"label":"attack","confidence":0.8,"reason":"max 20 words"}}
"""


def map_label_to_binary(label: str) -> str:
    label_clean = str(label).strip().lower()
    if label_clean == "background":
        return "normal"
    return "attack"


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


def append_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


def run_label(args: argparse.Namespace) -> int:
    return args.limit_records if args.limit_records is not None else args.sample_size


def prediction_file(output_dir: Path, sample_size: int, model_name: str) -> Path:
    return output_dir / f"predictions_{sample_size}_{model_name}.jsonl"


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
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


def normalize_json_like_text(text: str) -> str:
    """Normalize common LLM formatting before JSON parsing."""
    raw = (text or "").strip()
    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    raw = raw.replace("\ufeff", "").strip()
    return raw


def parse_model_response(text: str) -> Tuple[str, Optional[float], str]:
    """
    Robust parser for all providers, especially Gemini.

    Accepts:
    - clean JSON
    - JSON inside Markdown fences
    - JSON with prefix/suffix text
    - JSON array containing one object
    - single-word fallback: attack / normal
    """
    raw = normalize_json_like_text(text)

    candidates: List[str] = []
    if raw:
        candidates.append(raw)

    # Extract JSON object if model added text before/after it.
    obj_start = raw.find("{")
    obj_end = raw.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(raw[obj_start:obj_end + 1])

    # Extract first JSON array if returned as [{...}].
    arr_start = raw.find("[")
    arr_end = raw.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(raw[arr_start:arr_end + 1])

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not an object")

            label = str(data.get("label", "")).strip().lower()
            confidence = data.get("confidence", None)
            reason = str(data.get("reason", "")).strip()

            # Allow common synonyms but normalize final output.
            if label in {"benign", "normal traffic", "background"}:
                label = "normal"
            elif label in {"malicious", "anomaly", "suspicious"}:
                label = "attack"

            if label not in {"attack", "normal"}:
                raise ValueError(f"Invalid label: {label}")

            if confidence is not None:
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    confidence = None

            if not reason:
                reason = "valid_json_no_reason"

            return label, confidence, reason[:300]

        except Exception as exc:
            last_error = exc

    cleaned = raw.lower().strip().strip(".:;!\\\"'")
    if cleaned in {"attack", "normal"}:
        return cleaned, None, "single-word fallback"

    raise ValueError(f"Could not parse valid prediction. Last error={last_error}; raw={raw[:300]}")

def sleep_with_backoff(attempt: int, base_sleep: float = 2.0) -> None:
    delay = base_sleep * (2 ** (attempt - 1))
    delay += random.uniform(0, 0.5)
    time.sleep(delay)


def post_with_retries(
    url: str,
    headers: Optional[Dict[str, str]],
    body: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> requests.Response:
    retry_statuses = {429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)

            if response.status_code in retry_statuses:
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
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


def call_claude(prompt: str, api_key: str, model_id: str, timeout: int, max_retries: int) -> Dict[str, Any]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 200,
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


def call_chatgpt(prompt: str, api_key: str, model_id: str, timeout: int, max_retries: int) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 200,
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


def call_gemini(prompt: str, api_key: str, model_id: str, timeout: int, max_retries: int) -> Dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    t0 = time.time()
    r = post_with_retries(url, None, body, timeout, max_retries)
    latency = round(time.time() - t0, 3)

    data = r.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:1000]}")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()

    if not text:
        raise RuntimeError(
            "Gemini returned empty text "
            f"finishReason={finish_reason}; response={json.dumps(data)[:1000]}"
        )

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


def call_deepseek(prompt: str, api_key: str, model_id: str, timeout: int, max_retries: int) -> Dict[str, Any]:
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 200,
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
        "recall_sensitivity": round(recall, 6),
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


def print_checkpoint(
    completed_records: int,
    total_records: int,
    started_at: float,
    model_stats: Dict[str, Dict[str, Any]],
    monitoring_path: Optional[Path] = None,
) -> None:
    elapsed = time.time() - started_at
    pct = (completed_records / total_records) * 100 if total_records else 0
    done = int((pct / 100) * 30)
    bar = "█" * done + "░" * (30 - done)

    eta = 0
    if completed_records > 0:
        eta = (elapsed / completed_records) * (total_records - completed_records)

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append(f"CHECKPOINT: {completed_records}/{total_records} [{bar}] {pct:.1f}%")
    lines.append(f"Elapsed   : {dt.timedelta(seconds=int(elapsed))}")
    lines.append(f"ETA       : {dt.timedelta(seconds=int(eta))}")
    lines.append("-" * 100)
    lines.append(
        f"{'Model':<14} {'Calls':>7} {'Errors':>7} {'Skipped':>8} {'Attack':>8} {'Normal':>8} "
        f"{'AvgConf':>8} {'InTok':>10} {'OutTok':>8} {'Cost($)':>10} {'AvgLat':>8}"
    )
    lines.append("-" * 100)

    total_cost = 0.0
    for model_name, stat in model_stats.items():
        calls = stat["calls"]
        errors = stat["errors"]
        skipped = stat.get("skipped_resume", 0)
        in_tok = stat["input_tokens"]
        out_tok = stat["output_tokens"]
        cost = estimate_cost(model_name, in_tok, out_tok)
        total_cost += cost
        avg_lat = stat["total_latency"] / calls if calls else 0.0
        avg_conf = stat["confidence_sum"] / stat["confidence_count"] if stat["confidence_count"] else 0.0

        lines.append(
            f"{model_name:<14} {calls:>7} {errors:>7} {skipped:>8} "
            f"{stat['pred_attack']:>8} {stat['pred_normal']:>8} {avg_conf:>8.3f} "
            f"{in_tok:>10,} {out_tok:>8,} {cost:>10.6f} {avg_lat:>8.3f}"
        )

    lines.append("-" * 100)
    lines.append(f"Current estimated total cost: ${total_cost:.6f}")
    lines.append("=" * 100)

    msg = "\n".join(lines)
    print(msg, flush=True)
    if monitoring_path:
        append_log(monitoring_path, msg)


def write_text_report(
    path: Path,
    args: argparse.Namespace,
    attack_count: int,
    normal_count: int,
    model_stats: Dict[str, Dict[str, Any]],
    all_metrics: Dict[str, Dict[str, Any]],
    total_elapsed: float,
    monitoring_path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        W = f.write
        W("UGR16 Multi-Model LLM Evaluation Report — Final Stable Version V6\n")
        W("=" * 80 + "\n\n")
        W(f"Generated at       : {now_iso()}\n")
        W("Task               : Binary classification attack/normal\n")
        W(f"Base sample size   : {args.sample_size}\n")
        W(f"Processed records  : {run_label(args)}\n")
        W(f"API input file     : {args.api_file}\n")
        W(f"Eval file          : {args.eval_file}\n")
        W(f"Output directory   : {args.output_dir}\n")
        W(f"Monitoring log     : {monitoring_path}\n")
        W(f"Models             : {', '.join(args.models)}\n")
        W(f"Total runtime      : {dt.timedelta(seconds=int(total_elapsed))}\n")
        W(f"Ground truth attack: {attack_count}\n")
        W(f"Ground truth normal: {normal_count}\n\n")

        W("Method summary\n")
        W("-" * 80 + "\n")
        W(
            "Each JSONL record was sent independently to the selected pretrained LLM APIs. "
            "The prompt asked for a binary classification using only the provided NetFlow-style features. "
            "The model response was required to be JSON with label, confidence, and reason. "
            "A robust parser extracted valid JSON even when a model returned extra text or Markdown formatting. "
            "Predictions were joined with the evaluation JSONL file using record_id.\n\n"
        )

        W("Per-API behaviour and processing summary\n")
        W("-" * 80 + "\n")

        for model_name, stat in model_stats.items():
            cost = estimate_cost(model_name, stat["input_tokens"], stat["output_tokens"])
            avg_lat = stat["total_latency"] / stat["calls"] if stat["calls"] else 0.0
            avg_conf = stat["confidence_sum"] / stat["confidence_count"] if stat["confidence_count"] else 0.0
            min_lat = stat["min_latency"] if stat["min_latency"] is not None else 0.0
            max_lat = stat["max_latency"] if stat["max_latency"] is not None else 0.0

            W(f"\n{model_name.upper()}\n")
            W(f"  Model ID               : {stat['model_id']}\n")
            W(f"  Environment key        : {stat['env_key']}\n")
            W(f"  Successful calls        : {stat['calls']}\n")
            W(f"  Errors                  : {stat['errors']}\n")
            W(f"  Skipped by resume       : {stat.get('skipped_resume', 0)}\n")
            W(f"  Attack predictions      : {stat['pred_attack']}\n")
            W(f"  Normal predictions      : {stat['pred_normal']}\n")
            W(f"  Invalid/Error rows      : {stat['pred_error']}\n")
            W(f"  Average confidence      : {avg_conf:.4f}\n")
            W(f"  Low confidence (<0.50)  : {stat['low_confidence']}\n")
            W(f"  Input tokens            : {stat['input_tokens']}\n")
            W(f"  Output tokens           : {stat['output_tokens']}\n")
            W(f"  Avg latency             : {avg_lat:.3f} seconds\n")
            W(f"  Min latency             : {min_lat:.3f} seconds\n")
            W(f"  Max latency             : {max_lat:.3f} seconds\n")
            W(f"  Estimated cost          : ${cost:.6f}\n")

        W("\nEvaluation metrics by API\n")
        W("-" * 80 + "\n")
        for model_name, m in all_metrics.items():
            W(f"\n{model_name.upper()}\n")
            for k, v in m.items():
                W(f"  {k:<24}: {v}\n")

        # Comparison summary
        if all_metrics and model_stats:
            W("\nComparison summary\n")
            W("-" * 80 + "\n")

            def best(metric, reverse=True):
                return sorted(
                    all_metrics.items(),
                    key=lambda x: x[1].get(metric, 0),
                    reverse=reverse
                )[0][0]

            fastest = sorted(
                model_stats.items(),
                key=lambda x: (x[1]['total_latency'] / x[1]['calls'])
                if x[1]['calls'] else 999.0
            )[0][0]

            cheapest = sorted(
                model_stats.items(),
                key=lambda x: estimate_cost(
                    x[0], x[1]['input_tokens'], x[1]['output_tokens']
                )
            )[0][0]

            fewest_errors = sorted(
                model_stats.items(),
                key=lambda x: x[1]['errors']
            )[0][0]

            most_conservative = sorted(
                model_stats.items(),
                key=lambda x: x[1]['pred_normal'],
                reverse=True
            )[0][0]

            most_aggressive = sorted(
                model_stats.items(),
                key=lambda x: x[1]['pred_attack'],
                reverse=True
            )[0][0]

            W(f"  Best F1 score        : {best('f1')}\n")
            W(f"  Best precision       : {best('precision')}\n")
            W(f"  Best recall          : {best('recall_sensitivity')}\n")
            W(f"  Best accuracy        : {best('accuracy')}\n")
            W(f"  Best specificity     : {best('specificity')}\n")
            W(f"  Lowest FPR           : {best('false_positive_rate', reverse=False)}\n")
            W(f"  Lowest FNR           : {best('false_negative_rate', reverse=False)}\n")
            W(f"  Fastest model        : {fastest}\n")
            W(f"  Lowest cost          : {cheapest}\n")
            W(f"  Fewest errors        : {fewest_errors}\n")
            W(f"  Most conservative    : {most_conservative}  (most normal predictions)\n")
            W(f"  Most aggressive      : {most_aggressive}  (most attack predictions)\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UGR16 multi-model LLM binary evaluation script V6 final stable.")

    parser.add_argument("--sample-dir", default="/data/llm_samples_v10")
    parser.add_argument("--output-dir", default="/data/llm_eval_results_v6")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--api-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["chatgpt", "gemini", "deepseek", "claude"],
        choices=list(DEFAULT_MODEL_CONFIG.keys()),
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-between-calls", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-records", type=int, default=None)

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    args.output_dir = Path(args.output_dir)
    args.api_file = Path(args.api_file) if args.api_file else sample_dir / f"sample_{args.sample_size}_api.jsonl"
    args.eval_file = Path(args.eval_file) if args.eval_file else sample_dir / f"sample_{args.sample_size}_eval.jsonl"

    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_size = run_label(args)

    monitoring_path = args.output_dir / f"samples_monitoring_{effective_size}.log"
    errors_path = args.output_dir / f"samples_errors_{effective_size}.jsonl"

    append_log(monitoring_path, "=" * 100)
    append_log(monitoring_path, "UGR16 LLM EVALUATION MONITORING LOG — V6")
    append_log(monitoring_path, "=" * 100)
    append_log(monitoring_path, f"Started at       : {now_iso()}")
    append_log(monitoring_path, f"Models           : {', '.join(args.models)}")
    append_log(monitoring_path, f"API file         : {args.api_file}")
    append_log(monitoring_path, f"Eval file        : {args.eval_file}")
    append_log(monitoring_path, f"Output dir       : {args.output_dir}")
    append_log(monitoring_path, f"Effective records: {effective_size}")

    print("=" * 100, flush=True)
    print("UGR16 LLM Evaluation — V6 Final Stable", flush=True)
    print(f"Started     : {now_iso()}", flush=True)
    print(f"Models      : {', '.join(args.models)}", flush=True)
    print(f"API file    : {args.api_file}", flush=True)
    print(f"Eval file   : {args.eval_file}", flush=True)
    print(f"Output dir  : {args.output_dir}", flush=True)
    print(f"Monitor log : {monitoring_path}", flush=True)
    print("=" * 100, flush=True)

    if not args.api_file.exists():
        raise FileNotFoundError(f"API file not found: {args.api_file}")
    if not args.eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    api_keys = {}
    missing_keys = []

    for model_name in args.models:
        env_key = DEFAULT_MODEL_CONFIG[model_name]["env_key"]
        api_key = os.getenv(env_key, "").strip()
        if not api_key:
            missing_keys.append(f"{model_name}: missing {env_key}")
        api_keys[model_name] = api_key

    if missing_keys:
        for item in missing_keys:
            print(f"[MISSING_KEY] {item}", flush=True)
            append_log(monitoring_path, f"[MISSING_KEY] {item}")
        raise SystemExit("Set the required environment variables before running.")

    api_records = read_jsonl(args.api_file)
    if args.limit_records is not None:
        api_records = api_records[: args.limit_records]

    total_records = len(api_records)
    processed_record_ids = {str(r.get("record_id", "")) for r in api_records if r.get("record_id")}

    print(f"\nLoaded API records: {total_records:,}", flush=True)
    append_log(monitoring_path, f"Loaded API records: {total_records}")

    eval_records = read_jsonl(args.eval_file)
    eval_lookup = {}

    for rec in eval_records:
        rid = str(rec.get("record_id", ""))
        if rid and rid in processed_record_ids:
            eval_lookup[rid] = map_label_to_binary(str(rec.get("label", "")))

    attack_count = sum(1 for v in eval_lookup.values() if v == "attack")
    normal_count = sum(1 for v in eval_lookup.values() if v == "normal")

    print(f"Aligned ground truth: {attack_count:,} attack, {normal_count:,} normal", flush=True)
    append_log(monitoring_path, f"Aligned ground truth: {attack_count} attack, {normal_count} normal")

    if len(eval_lookup) != total_records:
        warning = (
            f"[WARNING] Eval alignment mismatch: api_records={total_records}, "
            f"matched_eval_records={len(eval_lookup)}"
        )
        print(warning, flush=True)
        append_log(monitoring_path, warning)

    if args.no_resume:
        for model_name in args.models:
            pfile = prediction_file(args.output_dir, effective_size, model_name)
            if pfile.exists():
                pfile.unlink()
        if errors_path.exists():
            errors_path.unlink()

    existing_by_model = {}
    for model_name in args.models:
        existing_by_model[model_name] = load_existing_predictions(
            prediction_file(args.output_dir, effective_size, model_name)
        )

    model_stats = {
        model_name: {
            "calls": 0,
            "errors": 0,
            "skipped_resume": len(existing_by_model[model_name]),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_latency": 0.0,
            "min_latency": None,
            "max_latency": None,
            "model_id": DEFAULT_MODEL_CONFIG[model_name]["model"],
            "env_key": DEFAULT_MODEL_CONFIG[model_name]["env_key"],
            "pred_attack": 0,
            "pred_normal": 0,
            "pred_error": 0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
            "low_confidence": 0,
        }
        for model_name in args.models
    }

    started_at = time.time()

    for idx, original_record in enumerate(api_records, start=1):
        record_id = str(original_record.get("record_id", ""))
        record_pct = (idx / total_records) * 100 if total_records else 0

        print(f"\n[RECORD] {idx}/{total_records} ({record_pct:.1f}%) record_id={record_id}", flush=True)
        append_log(
            monitoring_path,
            f"[RECORD_START] index={idx}/{total_records} pct={record_pct:.1f} record_id={record_id}",
        )

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
                print(f"  [SKIP] {model_name:<12} already exists from resume", flush=True)
                continue

            config = DEFAULT_MODEL_CONFIG[model_name]
            model_id = config["model"]
            caller = CALLERS[model_name]

            print(f"  [CALL] {model_name:<12} model={model_id}", flush=True)
            append_log(monitoring_path, f"[CALL] record_index={idx} record_id={record_id} model={model_name}")

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

                append_jsonl(prediction_file(args.output_dir, effective_size, model_name), row)
                existing_by_model[model_name][record_id] = row

                stat = model_stats[model_name]
                stat["calls"] += 1
                stat["input_tokens"] += result["input_tokens"]
                stat["output_tokens"] += result["output_tokens"]
                stat["total_latency"] += result["latency_s"]
                stat["min_latency"] = result["latency_s"] if stat["min_latency"] is None else min(stat["min_latency"], result["latency_s"])
                stat["max_latency"] = result["latency_s"] if stat["max_latency"] is None else max(stat["max_latency"], result["latency_s"])

                if result["prediction"] == "attack":
                    stat["pred_attack"] += 1
                elif result["prediction"] == "normal":
                    stat["pred_normal"] += 1

                if result["confidence"] is not None:
                    conf = float(result["confidence"])
                    stat["confidence_sum"] += conf
                    stat["confidence_count"] += 1
                    if conf < 0.50:
                        stat["low_confidence"] += 1

                print(
                    f"  [OK]   {model_name:<12} pred={result['prediction']:<6} "
                    f"conf={result['confidence']} latency={result['latency_s']}s "
                    f"tokens={result['input_tokens']}/{result['output_tokens']}",
                    flush=True,
                )

                append_log(
                    monitoring_path,
                    f"[OK] record_index={idx} record_id={record_id} model={model_name} "
                    f"prediction={result['prediction']} confidence={result['confidence']} "
                    f"latency_s={result['latency_s']}",
                )

            except Exception as exc:
                model_stats[model_name]["errors"] += 1
                model_stats[model_name]["pred_error"] += 1

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
                append_jsonl(
                    prediction_file(args.output_dir, effective_size, model_name),
                    {
                        **error_row,
                        "prediction": "error",
                        "confidence": None,
                        "reason": "api_or_parse_error",
                        "raw_response": str(exc),
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency_s": 0,
                    },
                )

                print(f"  [ERROR] {model_name:<12} {str(exc)[:180]}", flush=True)
                append_log(monitoring_path, f"[ERROR] record_index={idx} model={model_name} error={str(exc)}")

            if args.sleep_between_calls > 0:
                time.sleep(args.sleep_between_calls)

        if idx % args.checkpoint_every == 0 or idx == total_records:
            print_checkpoint(idx, total_records, started_at, model_stats, monitoring_path)

    all_metrics = {}

    for model_name in args.models:
        preds = list(
            load_existing_predictions(
                prediction_file(args.output_dir, effective_size, model_name)
            ).values()
        )
        all_metrics[model_name] = compute_metrics(preds, eval_lookup)

    total_elapsed = time.time() - started_at

    print("\n" + "=" * 100, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 100, flush=True)
    print(
        f"{'Model':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'FPR':>10} {'FNR':>10} {'Accuracy':>10} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}",
        flush=True,
    )
    print("-" * 100, flush=True)

    for model_name, m in all_metrics.items():
        print(
            f"{model_name:<14} {m['precision']:>10.4f} {m['recall_sensitivity']:>10.4f} "
            f"{m['f1']:>10.4f} {m['false_positive_rate']:>10.4f} "
            f"{m['false_negative_rate']:>10.4f} {m['accuracy']:>10.4f} "
            f"{m['TP']:>6} {m['FP']:>6} {m['TN']:>6} {m['FN']:>6}",
            flush=True,
        )

    print("=" * 100, flush=True)
    print(f"Total runtime: {dt.timedelta(seconds=int(total_elapsed))}", flush=True)

    report = {
        "script": "ugr16_llm_eval_v6_final.py",
        "version": "V6-final-stable-with-comparison-section",
        "generated_at": now_iso(),
        "task": "binary_classification_attack_normal",
        "method": {
            "description": (
                "Independent per-record LLM classification using pretrained APIs. "
                "Predictions are joined with ground truth by record_id. "
                "A robust parser extracts valid JSON from clean JSON, prefixed JSON, "
                "or Markdown fenced JSON."
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
        "base_sample_size": args.sample_size,
        "processed_records": effective_size,
        "limit_records": args.limit_records,
        "models": {
            model_name: {
                "model_id": DEFAULT_MODEL_CONFIG[model_name]["model"],
                "provider": DEFAULT_MODEL_CONFIG[model_name]["provider"],
            }
            for model_name in args.models
        },
        "ground_truth_aligned": {
            "attack": attack_count,
            "normal": normal_count,
            "matched_eval_records": len(eval_lookup),
        },
        "runtime": {
            "total_seconds": round(total_elapsed, 3),
            "total_hms": str(dt.timedelta(seconds=int(total_elapsed))),
        },
        "model_stats": model_stats,
        "metrics": all_metrics,
        "monitoring_log": str(monitoring_path),
        "cost_note": "Cost values are estimates only. Update prices from official provider pages before thesis submission.",
    }

    json_report_path = args.output_dir / f"samples_evaluation_report_{effective_size}.json"
    txt_report_path = args.output_dir / f"samples_evaluation_report_{effective_size}.txt"

    with json_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_text_report(
        txt_report_path,
        args,
        attack_count,
        normal_count,
        model_stats,
        all_metrics,
        total_elapsed,
        monitoring_path,
    )

    append_log(monitoring_path, "-" * 100)
    append_log(monitoring_path, f"Finished at : {now_iso()}")
    append_log(monitoring_path, f"JSON report : {json_report_path}")
    append_log(monitoring_path, f"TXT report  : {txt_report_path}")
    append_log(monitoring_path, f"Errors file : {errors_path}")
    append_log(monitoring_path, "Evaluation complete.")

    print("\nReports saved:", flush=True)
    print(f"  JSON: {json_report_path}", flush=True)
    print(f"  TXT : {txt_report_path}", flush=True)
    print(f"  LOG : {monitoring_path}", flush=True)
    print(f"  ERR : {errors_path}", flush=True)
    print("\nEvaluation complete.", flush=True)


if __name__ == "__main__":
    main()