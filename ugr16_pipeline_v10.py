# ============================================================
# UGR16 → Suspect Selection → Sampling → JSONL + Report
# Version 10 — stratified sampling
#
# Built on v9. One change only: Step 4 replaced with
# stratified sampling by label class.
#
# Why stratified sampling:
#   v9 pure random sampling drew 5,000 records from a pool
#   that is 99.98% background. Result: 4,999 background,
#   1 blacklist, 0 sshscan. With 1 attack record, precision
#   recall and F1 are not computable — evaluation is invalid.
#
#   v10 samples each label class separately then unions:
#     background     : 4,700 records (random, seed=42)
#     blacklist      :   290 records (random, seed=42)
#     anomaly-sshscan:     4 records (all available — only 4 exist)
#     Total          : 4,994 records
#
#   The 6-record shortfall from 5,000 is because sshscan has
#   only 4 records in the suspect pool. Documented in report.
#
# Confirmed schema (from live inspection):
#   - 13 columns, all string, positional names _c0–_c12
#   - protocol stored as text: "TCP","UDP","ICMP"...
#   - All numeric fields stored as strings: "13","2913"...
#
# spark-submit command for Azure (8 CPU / 64 GB):
#   /opt/spark/bin/spark-submit \
#     --driver-memory  10g \
#     --executor-memory 42g \
#     /data/ugr16_pipeline_v10.py
# ============================================================

import os
import glob
import shutil
import time
import json
import math
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, rand, row_number,
    sha2, concat_ws, to_json, struct,
    coalesce, count as spark_count,
    min as spark_min, max as spark_max,
    upper, sum as spark_sum
)
from pyspark.sql.window import Window


# ============================================================
# Configuration
# ============================================================

INPUT_PATH  = "/data/parquet"
OUTPUT_BASE = "/data/llm_samples_v10"

RANDOM_SEED        = 42
SHUFFLE_PARTITIONS = 24

# Stratified sample targets
N_BACKGROUND = 4700
N_BLACKLIST  = 290
# anomaly-sshscan: take all available (only 4 in suspect pool)

COMPUTE_SUSPECT_POOL_DISTRIBUTION = True

os.makedirs(OUTPUT_BASE, exist_ok=True)


# ============================================================
# Column definitions
# ============================================================

API_COLUMNS = [
    "record_id", "sample_rank",
    "timestamp", "duration",
    "src_ip",    "dst_ip",
    "src_port",  "dst_port",
    "protocol",  "flags",
    "fwd_pkts",  "bwd_pkts",
    "total_pkts","total_bytes"
]

EVAL_COLUMNS = API_COLUMNS + ["suspect_reason", "label"]


# ============================================================
# Helpers
# ============================================================

def write_single_jsonl(df, temp_dir, final_file):
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(final_file):
            os.remove(final_file)

        df.coalesce(1).write.mode("overwrite").text(temp_dir)

        parts = glob.glob(os.path.join(temp_dir, "part-*"))
        if not parts:
            raise RuntimeError(
                f"No part file in {temp_dir} — DataFrame may be empty."
            )
        shutil.move(parts[0], final_file)
        shutil.rmtree(temp_dir)

    except Exception as exc:
        raise RuntimeError(
            f"write_single_jsonl failed for {final_file}"
        ) from exc


def to_jsonl(df_in, columns):
    return df_in.select(
        to_json(struct(*[col(c) for c in columns])).alias("value")
    )


def safe_pct(numerator, denominator, decimals=4):
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100, decimals)


def label_distribution(df_in):
    rows = (
        df_in.groupBy("label")
             .agg(spark_count("*").alias("n"))
             .orderBy(col("n").desc())
             .collect()
    )
    return {str(r["label"]): int(r["n"]) for r in rows}


def ci_margin(n, z=1.96, p=0.5):
    if n <= 0:
        return None
    return round(z * math.sqrt(p * (1 - p) / n) * 100, 2)


# ============================================================
# Methodology notes
# ============================================================

METHODOLOGY_NOTES = {

    "stratified_sampling_rationale": (
        "Pure random sampling (v9) drew 5,000 records from a suspect "
        "pool that is 99.98% background, producing 4,999 background "
        "and 1 blacklist record. With 1 attack record, per-class "
        "precision, recall, and F1 are not computable. "
        "v10 applies stratified sampling by label class to ensure "
        "minimum attack class representation while preserving background "
        "dominance. Background constitutes 94.1% of the stratified "
        "sample, consistent with its real-world prevalence. "
        "Oversampling of minority attack classes is declared explicitly "
        "as a methodological choice required for valid evaluation."
    ),

    "stratified_sample_targets": {
        "background":       N_BACKGROUND,
        "blacklist":        N_BLACKLIST,
        "anomaly-sshscan":  "all available (4 records in suspect pool)"
    },

    "threshold_justification": {
        "total_pkts_gt_1000": (
            "Threshold of 1,000 packets selected as a design choice "
            "to capture flows with sustained traffic volume. "
            "Exact percentile to be computed from full dataset statistics."
        ),
        "total_bytes_gt_1000000": (
            "Threshold of 1,000,000 bytes (1 MB) selected to capture "
            "high-volume flows. Exact percentile to be computed from data."
        ),
        "sensitive_ports_22_23_445_3389": (
            "Ports selected based on common attack surface: "
            "22 (SSH), 23 (Telnet), 445 (SMB), 3389 (RDP). "
            "Citation required: SANS Institute Top Attacked Ports."
        ),
        "icmp_traffic": (
            "ICMP included for ping flood and reconnaissance detection. "
            "upper(protocol) == 'ICMP' handles case inconsistency. "
            "Citation: Maciá-Fernández et al. (2018) UGR16 dataset paper."
        )
    },

    "sample_size_rationale": {
        "formula":   "margin_of_error = z * sqrt(p*(1-p)/n)",
        "computed_margins": {
            "n_1000": f"±{ci_margin(1000)}%",
            "n_3000": f"±{ci_margin(3000)}%",
            "n_5000": f"±{ci_margin(5000)}%",
        },
        "interpretation": (
            f"At n=5000, margin of error = ±{ci_margin(5000)}% (95% CI). "
            f"At n=1000, margin of error = ±{ci_margin(1000)}% (95% CI). "
            "Nested design allows sensitivity analysis across scales."
        )
    },

    "limitations": [
        (
            "Temporal stratification: random sampling applied without "
            "time-window stratification. Declared as threat to external validity."
        ),
        (
            "anomaly-spam absent: spam flows do not trigger any of the "
            "four suspect rules. Spam detection is excluded from evaluation scope."
        ),
        (
            "anomaly-sshscan: only 4 records available in suspect pool. "
            "Statistical conclusions about sshscan detection are not possible."
        ),
        (
            "Threshold selection: thresholds chosen as design decisions, "
            "not derived from computed percentiles. Sensitivity analysis recommended."
        ),
        (
            "Minority class oversampling: blacklist and sshscan are "
            "overrepresented relative to their natural prevalence. "
            "Results reflect evaluation performance, not real-world detection rates."
        )
    ]
}


# ============================================================
# Report skeleton
# ============================================================

job_start = time.time()

report = {
    "job_name":    "UGR16_Reproducible_Sampling_v10",
    "version":     "10.0",
    "start_time":  datetime.now().isoformat(),
    "input_path":  INPUT_PATH,
    "output_base": OUTPUT_BASE,
    "confirmed_schema": {
        "source_dtypes":   "all string (confirmed from live inspection)",
        "column_names":    "_c0 to _c12 positional, renamed via toDF()",
        "protocol_values": "text: ICMP, TCP, UDP, IPv6, GRE, ESP..."
    },
    "methodology": {
        "sampling_strategy": "stratified by label class",
        "suspect_rules": [
            "total_pkts > 1000            => high_packet_count",
            "total_bytes > 1000000        => high_bytes",
            "dst_port in [22,23,445,3389] => sensitive_port",
            "upper(protocol) == ICMP      => icmp_traffic"
        ],
        "random_seed":    RANDOM_SEED,
        "sample_design":  "Stratified: 4700 background + 290 blacklist + all sshscan",
        "nested_design":  "Nested: 1000 ⊂ 3000 ⊂ 5000 within stratified pool",
        "cast_strategy": (
            "All columns are string. Numeric casts applied inside "
            "each when() expression to guarantee correct typed comparison."
        ),
        "label_policy": (
            "API files: no label, no suspect_reason. "
            "Eval files: label + suspect_reason for metric calculation."
        ),
        "notes": METHODOLOGY_NOTES
    },
    "stage_times_seconds": {}
}


# ============================================================
# Spark session
# ============================================================

spark = (
    SparkSession.builder
    .appName("UGR16_Reproducible_Sampling_v10")
    .config("spark.sql.shuffle.partitions",      str(SHUFFLE_PARTITIONS))
    .config("spark.sql.files.maxPartitionBytes", "128m")
    .getOrCreate()
)

report["methodology"]["spark_version"]       = spark.version
report["methodology"]["default_parallelism"] = spark.sparkContext.defaultParallelism

print("=== Spark started ===")
print(f"    Version            : {spark.version}")
print(f"    Default parallelism: {spark.sparkContext.defaultParallelism}")
print(f"    Shuffle partitions : {SHUFFLE_PARTITIONS}")


# ============================================================
# RULES — defined after Spark starts (col() requires SparkSession)
# ============================================================

RULES = [
    {
        "name":      "high_packet_count",
        "condition": col("total_pkts").cast("long") > 1000,
        "label":     "total_pkts > 1000"
    },
    {
        "name":      "high_bytes",
        "condition": col("total_bytes").cast("long") > 1000000,
        "label":     "total_bytes > 1000000"
    },
    {
        "name":      "sensitive_port",
        "condition": col("dst_port").cast("int").isin(22, 23, 445, 3389),
        "label":     "dst_port in [22,23,445,3389]"
    },
    {
        "name":      "icmp_traffic",
        "condition": upper(col("protocol")) == "ICMP",
        "label":     "upper(protocol) == ICMP"
    },
]


try:

    # ============================================================
    # Step 1 — Load and rename columns
    # ============================================================

    t0 = time.time()

    df = spark.read.parquet(INPUT_PATH)

    if len(df.columns) != 13:
        raise ValueError(
            f"Expected 13 columns, found {len(df.columns)}: {df.columns}"
        )

    df = df.toDF(
        "timestamp", "duration",
        "src_ip",    "dst_ip",
        "src_port",  "dst_port",
        "protocol",  "flags",
        "fwd_pkts",  "bwd_pkts",
        "total_pkts","total_bytes",
        "label"
    )

    total_records = df.count()

    report["total_records"] = total_records
    report["stage_times_seconds"]["load_and_rename"] = round(time.time() - t0, 2)

    print(f"\nTotal records: {total_records:,}")


    # ============================================================
    # Step 2a — Full dataset label distribution  [Addition A]
    # ============================================================

    t0 = time.time()

    print("Computing full dataset label distribution ...")
    full_label_dist = label_distribution(df)

    report["full_dataset_label_distribution"] = full_label_dist
    report["stage_times_seconds"]["full_label_distribution"] = (
        round(time.time() - t0, 2)
    )

    print("Full dataset label distribution:")
    for lbl, n in full_label_dist.items():
        print(f"  {lbl:<30}: {n:>12,}  ({safe_pct(n, total_records, 2):.2f}%)")


    # ============================================================
    # Step 2b — Rule contribution count  [Addition C]
    # Single pass — 1 scan of 862M records for all 4 rules
    # ============================================================

    t0 = time.time()

    print("\nComputing rule contributions (single pass) ...")

    df_flagged = df.select(
        when(RULES[0]["condition"], lit(1)).otherwise(lit(0)).alias("f_pkts"),
        when(RULES[1]["condition"], lit(1)).otherwise(lit(0)).alias("f_bytes"),
        when(RULES[2]["condition"], lit(1)).otherwise(lit(0)).alias("f_port"),
        when(RULES[3]["condition"], lit(1)).otherwise(lit(0)).alias("f_icmp")
    )

    agg_row = df_flagged.agg(
        spark_sum("f_pkts").alias("high_packet_count"),
        spark_sum("f_bytes").alias("high_bytes"),
        spark_sum("f_port").alias("sensitive_port"),
        spark_sum("f_icmp").alias("icmp_traffic")
    ).collect()[0]

    rule_contributions = {}
    for rule in RULES:
        n = int(agg_row[rule["name"]] or 0)
        rule_contributions[rule["name"]] = {
            "condition":       rule["label"],
            "matched_records": n,
            "pct_of_full":     safe_pct(n, total_records, 4)
        }
        print(f"  {rule['name']:<25}: {n:>12,}  "
              f"({safe_pct(n, total_records, 2):.2f}%)")

    report["rule_contributions"] = rule_contributions
    report["stage_times_seconds"]["rule_contributions"] = (
        round(time.time() - t0, 2)
    )


    # ============================================================
    # Step 2c — Suspect selection + overlap  [Addition B + D]
    # ============================================================

    t0 = time.time()

    suspect_df = (
        df.withColumn(
            "suspect_reason",
            concat_ws(
                ",",
                *[when(r["condition"], lit(r["name"])) for r in RULES]
            )
        )
        .filter(col("suspect_reason") != "")
    )

    suspect_count = suspect_df.count()

    report["suspect_records"]       = suspect_count
    report["suspect_ratio_percent"] = safe_pct(suspect_count, total_records)
    report["stage_times_seconds"]["suspect_filter_and_count"] = (
        round(time.time() - t0, 2)
    )

    print(f"\nSuspect records: {suspect_count:,} "
          f"({report['suspect_ratio_percent']:.4f}%)")

    if suspect_count == 0:
        raise ValueError("Suspect filter returned 0 records.")

    # ── Suspect pool label distribution  [Addition B] ─────────
    t0 = time.time()

    if COMPUTE_SUSPECT_POOL_DISTRIBUTION:
        print("Computing suspect pool label distribution ...")
        suspect_label_dist = label_distribution(suspect_df)
    else:
        suspect_label_dist = "skipped (COMPUTE_SUSPECT_POOL_DISTRIBUTION=False)"

    report["suspect_pool_label_distribution"] = suspect_label_dist
    report["stage_times_seconds"]["suspect_label_distribution"] = (
        round(time.time() - t0, 2)
    )

    if isinstance(suspect_label_dist, dict):
        print("Suspect pool label distribution:")
        for lbl, n in suspect_label_dist.items():
            print(f"  {lbl:<30}: {n:>12,}  ({safe_pct(n, suspect_count, 2):.2f}%)")

    # ── Rule overlap distribution  [Addition D] ───────────────
    t0 = time.time()

    print("Computing rule overlap distribution ...")
    overlap_rows = (
        suspect_df
        .groupBy("suspect_reason")
        .agg(spark_count("*").alias("n"))
        .orderBy(col("n").desc())
        .collect()
    )
    rule_overlap = {
        r["suspect_reason"]: {
            "count":          int(r["n"]),
            "pct_of_suspect": safe_pct(int(r["n"]), suspect_count, 4)
        }
        for r in overlap_rows
    }

    report["rule_overlap_distribution"] = rule_overlap
    report["stage_times_seconds"]["rule_overlap"] = round(time.time() - t0, 2)


    # ============================================================
    # Step 3 — Stable record_id (SHA-256, null-safe)
    # ============================================================

    suspect_df = suspect_df.withColumn(
        "record_id",
        sha2(
            concat_ws(
                "|",
                coalesce(col("timestamp"),   lit("NULL")),
                coalesce(col("src_ip"),      lit("NULL")),
                coalesce(col("dst_ip"),      lit("NULL")),
                coalesce(col("src_port"),    lit("NULL")),
                coalesce(col("dst_port"),    lit("NULL")),
                coalesce(col("protocol"),    lit("NULL")),
                coalesce(col("duration"),    lit("NULL")),
                coalesce(col("total_pkts"),  lit("NULL")),
                coalesce(col("total_bytes"), lit("NULL"))
            ),
            256
        )
    )


    # ============================================================
    # Step 4 — Stratified sampling  ← KEY CHANGE FROM v9
    #
    # v9 problem: pure random from 99.98% background pool
    #   → 4,999 background, 1 attack → evaluation invalid
    #
    # v10 solution: sample each label class separately
    #   background     : 4,700 (random, seed=42)
    #   blacklist      :   290 (random, seed=42)
    #   anomaly-sshscan:     4 (all available)
    #   Total          : 4,994
    #
    # Same seed=42 used for each class independently.
    # _rand assigned per class before ordering to ensure
    # reproducibility within each stratum.
    # ============================================================

    t0 = time.time()

    print("\nApplying stratified sampling by label class ...")

    # ── Split suspect pool by label ───────────────────────────
    bg_df  = suspect_df.filter(col("label") == "background")
    bl_df  = suspect_df.filter(col("label") == "blacklist")
    ssh_df = suspect_df.filter(col("label") == "anomaly-sshscan")

    # ── Sample each class independently ──────────────────────
    bg_sample = (
        bg_df
        .withColumn("_rand", rand(seed=RANDOM_SEED))
        .orderBy(col("_rand"), col("record_id"))
        .limit(N_BACKGROUND)
        .drop("_rand")
        .cache()
    )
    bg_sample.count()
    print(f"  background sample  : {bg_sample.count():,}")

    bl_sample = (
        bl_df
        .withColumn("_rand", rand(seed=RANDOM_SEED))
        .orderBy(col("_rand"), col("record_id"))
        .limit(N_BLACKLIST)
        .drop("_rand")
        .cache()
    )
    bl_sample.count()
    print(f"  blacklist sample   : {bl_sample.count():,}")

    ssh_sample = ssh_df.cache()   # take all — only 4 exist
    ssh_count  = ssh_sample.count()
    print(f"  sshscan sample     : {ssh_count:,} (all available)")

    # ── Union all strata ──────────────────────────────────────
    combined = bg_sample.union(bl_sample).union(ssh_sample).cache()
    combined.count()

    bg_sample.unpersist()
    bl_sample.unpersist()
    ssh_sample.unpersist()

    # ── Assign sample_rank across combined pool ───────────────
    rank_window = Window.orderBy(col("timestamp"), col("record_id"))

    sample_5000 = (
        combined
        .withColumn("sample_rank", row_number().over(rank_window))
        .cache()
    )

    final_count = sample_5000.count()
    combined.unpersist()

    # ── Record stratified counts in report ───────────────────
    strat_dist = label_distribution(sample_5000)

    report["stratified_sample_counts"] = strat_dist
    report["stratified_sample_targets"] = {
        "background":        N_BACKGROUND,
        "blacklist":         N_BLACKLIST,
        "anomaly-sshscan":   "all available"
    }
    report["final_sample_count"]             = final_count
    report["sample_ratio_from_full_percent"] = safe_pct(final_count, total_records)
    report["stage_times_seconds"]["stratified_sampling"] = (
        round(time.time() - t0, 2)
    )

    print(f"\nStratified sample label distribution:")
    for lbl, n in strat_dist.items():
        print(f"  {lbl:<30}: {n:>6,}  ({safe_pct(n, final_count, 2):.2f}%)")
    print(f"Total: {final_count:,}")

    # ── record_id uniqueness check ────────────────────────────
    t_uid = time.time()
    unique_ids    = sample_5000.select("record_id").distinct().count()
    duplicate_ids = final_count - unique_ids

    report["unique_record_ids_sample_5000"]    = unique_ids
    report["duplicate_record_ids_sample_5000"] = duplicate_ids
    report["stage_times_seconds"]["record_id_uniqueness_check"] = (
        round(time.time() - t_uid, 2)
    )

    if duplicate_ids > 0:
        print(f"WARNING: {duplicate_ids:,} duplicate record_id(s). Logged.")
    else:
        print(f"record_id uniqueness: PASSED ({unique_ids:,} unique IDs)")


    # ============================================================
    # Step 5 — Nested sub-samples  (1000 ⊂ 3000 ⊂ 5000)
    #
    # Nested samples are drawn from the stratified pool.
    # sample_rank ordering is by timestamp + record_id so
    # smaller samples are coherent subsets of larger ones.
    # ============================================================

    t0 = time.time()

    sample_1000 = sample_5000.filter(col("sample_rank") <= 1000).cache()
    sample_3000 = sample_5000.filter(col("sample_rank") <= 3000).cache()

    count_1000 = sample_1000.count()
    count_3000 = sample_3000.count()
    count_5000 = final_count

    # Label distribution per nested sample
    dist_1000 = label_distribution(sample_1000)
    dist_3000 = label_distribution(sample_3000)

    report["nested_sample_counts"] = {
        "sample_1000": count_1000,
        "sample_3000": count_3000,
        "sample_5000": count_5000
    }
    report["nested_label_distributions"] = {
        "sample_1000": dist_1000,
        "sample_3000": dist_3000,
        "sample_5000": strat_dist
    }
    report["stage_times_seconds"]["nested_samples"] = round(time.time() - t0, 2)

    print(f"\n  sample_1000: {count_1000:,} — {dist_1000}")
    print(f"  sample_3000: {count_3000:,} — {dist_3000}")
    print(f"  sample_5000: {count_5000:,} — {strat_dist}")


    # ============================================================
    # Step 6 — Timestamp range  [Addition E]
    # ============================================================

    t0 = time.time()

    ts_row = sample_5000.select(
        spark_min("timestamp").alias("first"),
        spark_max("timestamp").alias("last")
    ).collect()[0]

    ts_range = {
        "first_timestamp": str(ts_row["first"]),
        "last_timestamp":  str(ts_row["last"]),
        "note": (
            "Timestamp range of stratified sample_5000. "
            "Random sampling without temporal stratification — "
            "does not guarantee uniform temporal distribution."
        )
    }

    report["sample_5000_timestamp_range"] = ts_range
    report["stage_times_seconds"]["timestamp_range"] = (
        round(time.time() - t0, 2)
    )

    print(f"\nTimestamp range:")
    print(f"  First: {ts_range['first_timestamp']}")
    print(f"  Last : {ts_range['last_timestamp']}")


    # ============================================================
    # Step 7 — Build JSONL DataFrames
    # ============================================================

    api_1000  = to_jsonl(sample_1000, API_COLUMNS)
    api_3000  = to_jsonl(sample_3000, API_COLUMNS)
    api_5000  = to_jsonl(sample_5000, API_COLUMNS)

    eval_1000 = to_jsonl(sample_1000, EVAL_COLUMNS)
    eval_3000 = to_jsonl(sample_3000, EVAL_COLUMNS)
    eval_5000 = to_jsonl(sample_5000, EVAL_COLUMNS)


    # ============================================================
    # Step 8 — Write JSONL files
    # ============================================================

    t0 = time.time()

    output_specs = [
        (api_1000,  "tmp_api_1000",  "sample_1000_api.jsonl"),
        (api_3000,  "tmp_api_3000",  "sample_3000_api.jsonl"),
        (api_5000,  "tmp_api_5000",  "sample_5000_api.jsonl"),
        (eval_1000, "tmp_eval_1000", "sample_1000_eval.jsonl"),
        (eval_3000, "tmp_eval_3000", "sample_3000_eval.jsonl"),
        (eval_5000, "tmp_eval_5000", "sample_5000_eval.jsonl"),
    ]

    output_paths = []
    for df_out, tmp_name, file_name in output_specs:
        tmp_path   = os.path.join(OUTPUT_BASE, tmp_name)
        final_path = os.path.join(OUTPUT_BASE, file_name)
        write_single_jsonl(df_out, tmp_path, final_path)
        output_paths.append(final_path)
        print(f"  Written: {file_name}")

    report["stage_times_seconds"]["write_files"] = round(time.time() - t0, 2)


    # ============================================================
    # Step 9 — Free memory
    # ============================================================

    sample_1000.unpersist()
    sample_3000.unpersist()
    sample_5000.unpersist()


    # ============================================================
    # Step 10 — Validate + report
    # ============================================================

    t0 = time.time()

    expected_lines = {
        "sample_1000_api.jsonl":  count_1000,
        "sample_3000_api.jsonl":  count_3000,
        "sample_5000_api.jsonl":  count_5000,
        "sample_1000_eval.jsonl": count_1000,
        "sample_3000_eval.jsonl": count_3000,
        "sample_5000_eval.jsonl": count_5000,
    }

    file_info    = {}
    missing_outs = []
    line_errors  = []

    for file_path in output_paths:
        name = os.path.basename(file_path)
        if os.path.exists(file_path):
            with open(file_path, "r") as fh:
                line_count = sum(1 for _ in fh)
            expected = expected_lines.get(name)
            ok = (line_count == expected)
            if not ok:
                line_errors.append(
                    f"{name}: expected {expected:,}, got {line_count:,}"
                )
            file_info[file_path] = {
                "size_bytes":     os.path.getsize(file_path),
                "size_mb":        round(os.path.getsize(file_path) / (1024**2), 4),
                "line_count":     line_count,
                "expected_lines": expected,
                "line_count_ok":  ok
            }
        else:
            missing_outs.append(file_path)

    if missing_outs:
        report["missing_output_files"] = missing_outs
        print(f"WARNING — missing files: {missing_outs}")

    if line_errors:
        raise ValueError(
            "Line count mismatch:\n  " + "\n  ".join(line_errors)
        )

    report["output_files"]                 = file_info
    report["output_line_count_validation"] = "PASSED"
    report["end_time"]                     = datetime.now().isoformat()
    report["processing_time_seconds"]      = round(time.time() - job_start, 2)
    report["processing_time_minutes"]      = round((time.time() - job_start) / 60, 2)
    report["stage_times_seconds"]["build_report"] = round(time.time() - t0, 2)

    print("\nLine-count validation: PASSED")


    # ── JSON report ──────────────────────────────────────────

    json_path = os.path.join(OUTPUT_BASE, "processing_report.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=4)


    # ── TXT report ───────────────────────────────────────────

    txt_path = os.path.join(OUTPUT_BASE, "processing_report.txt")
    with open(txt_path, "w") as fh:
        W = fh.write
        W("UGR16 Spark Sampling — Processing Report (v10)\n")
        W("=" * 54 + "\n\n")
        W(f"Job name        : {report['job_name']}\n")
        W(f"Version         : {report['version']}\n")
        W(f"Start time      : {report['start_time']}\n")
        W(f"End time        : {report['end_time']}\n")
        W(f"Processing time : {report['processing_time_minutes']} minutes\n\n")

        W("Infrastructure\n")
        W("-" * 40 + "\n")
        W(f"  Spark version      : {report['methodology']['spark_version']}\n")
        W(f"  Default parallelism: {report['methodology']['default_parallelism']}\n")
        W(f"  Shuffle partitions : {SHUFFLE_PARTITIONS}\n")
        W(f"  Random seed        : {RANDOM_SEED}\n\n")

        W("Dataset summary\n")
        W("-" * 40 + "\n")
        W(f"  Total records      : {total_records:,}\n")
        W(f"  Suspect records    : {suspect_count:,} "
          f"({report['suspect_ratio_percent']:.4f}%)\n")
        W(f"  Final sample       : {final_count:,}\n\n")

        W("Full dataset label distribution\n")
        W("-" * 40 + "\n")
        for lbl, n in full_label_dist.items():
            W(f"  {lbl:<30}: {n:>12,}  ({safe_pct(n, total_records, 2):.2f}%)\n")
        W("\n")

        W("Suspect pool label distribution\n")
        W("-" * 40 + "\n")
        if isinstance(suspect_label_dist, dict):
            for lbl, n in suspect_label_dist.items():
                W(f"  {lbl:<30}: {n:>12,}  ({safe_pct(n, suspect_count, 2):.2f}%)\n")
        W("\n")

        W("Stratified sample — label distribution\n")
        W("-" * 40 + "\n")
        W(f"  Sampling strategy: stratified by label class\n")
        W(f"  background target : {N_BACKGROUND:,}\n")
        W(f"  blacklist target  : {N_BLACKLIST:,}\n")
        W(f"  sshscan target    : all available (4)\n\n")
        for lbl, n in strat_dist.items():
            W(f"  {lbl:<30}: {n:>6,}  ({safe_pct(n, final_count, 2):.2f}%)\n")
        W("\n")

        W("Nested sample label distributions\n")
        W("-" * 40 + "\n")
        for sname, dist in report["nested_label_distributions"].items():
            W(f"  {sname}:\n")
            for lbl, n in dist.items():
                cnt = report["nested_sample_counts"][sname]
                W(f"    {lbl:<28}: {n:>6,}  ({safe_pct(n, cnt, 2):.2f}%)\n")
        W("\n")

        W("Rule contribution (single pass, full dataset)\n")
        W("-" * 40 + "\n")
        for rname, rinfo in rule_contributions.items():
            W(f"  {rname:<25}: {rinfo['matched_records']:>12,}  "
              f"({rinfo['pct_of_full']:.4f}%)\n")
        W("\n")

        W("Rule overlap distribution\n")
        W("-" * 40 + "\n")
        for reason, info in rule_overlap.items():
            W(f"  {reason:<45}: {info['count']:>10,}  "
              f"({info['pct_of_suspect']:.2f}%)\n")
        W("\n")

        W("Timestamp range (sample_5000)\n")
        W("-" * 40 + "\n")
        W(f"  First: {ts_range['first_timestamp']}\n")
        W(f"  Last : {ts_range['last_timestamp']}\n\n")

        W("Sample size confidence intervals\n")
        W("-" * 40 + "\n")
        ci = METHODOLOGY_NOTES["sample_size_rationale"]["computed_margins"]
        W(f"  n=1000 : ±{ci['n_1000']} (95% CI)\n")
        W(f"  n=3000 : ±{ci['n_3000']} (95% CI)\n")
        W(f"  n=5000 : ±{ci['n_5000']} (95% CI)\n\n")

        W("Stratified sampling rationale\n")
        W("-" * 40 + "\n")
        W(f"  {METHODOLOGY_NOTES['stratified_sampling_rationale']}\n\n")

        W("Declared limitations\n")
        W("-" * 40 + "\n")
        for i, lim in enumerate(METHODOLOGY_NOTES["limitations"], 1):
            W(f"  {i}. {lim}\n\n")

        W("Integrity checks\n")
        W("-" * 40 + "\n")
        dup_status = (
            "PASSED" if duplicate_ids == 0
            else f"WARNING — {duplicate_ids:,} duplicate(s)"
        )
        W(f"  record_id uniqueness : {unique_ids:,} unique — {dup_status}\n")
        W(f"  Line-count validation: {report['output_line_count_validation']}\n\n")

        W("Stage processing times\n")
        W("-" * 40 + "\n")
        for stage, secs in report["stage_times_seconds"].items():
            W(f"  {stage:<45}: {secs} s\n")
        W("\n")

        W("Output files\n")
        W("-" * 40 + "\n")
        for path, info in file_info.items():
            W(f"  {os.path.basename(path)}\n")
            W(f"    Lines   : {info['line_count']:,}\n")
            W(f"    Size MB : {info['size_mb']}\n")
        W("\n")


    print("\n=== Output files ===")
    for p in output_paths:
        print(f"  {p}")
    print(f"\n=== Reports ===")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print(f"\nTotal time: {report['processing_time_minutes']} minutes")


finally:
    spark.stop()
    print("\n=== Spark stopped ===")
    print("=== Done ===")
