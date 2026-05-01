# Intelligent Threat Detection Using Large Language Models
## in Apache Spark-Based Distributed Computing Environments

**PhD Research — University of Granada**
**Candidate:** Thamer Aldraiwa
**Supervisors:** Prof. Juan Carlos Gámez Granados & Prof. Antonio Mora García
**Date:** May 2026

---

## Overview

This repository contains the complete experimental pipeline for evaluating pretrained Large Language Models (LLMs) on network intrusion detection using the UGR'16 dataset.

The research investigates whether general-purpose LLMs — **Claude**, **ChatGPT**, **Gemini**, and **DeepSeek** — can classify network flows as **attack** or **normal** without any domain-specific training, using only raw NetFlow features.

---

## Dataset

| Property | Value |
|---|---|
| Dataset | UGR'16 — University of Granada Network Intrusion Dataset |
| Source | Maciá-Fernández et al. (2018) |
| Download URL | https://nesg.ugr.es/nesg-ugr16/download/attack/august/week4/august_week4_csv.tar.gz |
| Original CSV size | 77.6 GB |
| Parquet size | 14 GB (311 files, Snappy compressed) |
| Total records | 862,526,861 network flow records |
| Label classes | background, anomaly-spam, blacklist, anomaly-sshscan |

> **Note:** The dataset is not included in this repository due to its size (14 GB Parquet / 77.6 GB CSV). Use the download URL above and follow the acquisition steps in Section 3 below.

---

## Repository Structure

```
├── ugr16_pipeline_v9.py              # Diagnostic pipeline — revealed class imbalance
├── ugr16_pipeline_v10.py             # Production pipeline — stratified sampling
├── ugr16_llm_eval_v2.py             # LLM evaluation script v2
├── Parquet Conversion/               # CSV to Parquet conversion script
├── create_sample_5000_3000_1000_...  # Early sampling prototype (reference only)
├── UGR16 Spark Sampling — Processing Report  # v10 processing report
├── samples/                          # JSONL sample files (v10 output)
│   ├── sample_1000_api.jsonl         # 1,000 records — no label — sent to LLM
│   ├── sample_3000_api.jsonl         # 3,000 records — no label — sent to LLM
│   ├── sample_5000_api.jsonl         # 4,994 records — no label — sent to LLM
│   ├── sample_1000_eval.jsonl        # 1,000 records — with label — for metrics
│   ├── sample_3000_eval.jsonl        # 3,000 records — with label — for metrics
│   └── sample_5000_eval.jsonl        # 4,994 records — with label — for metrics
└── README.md
```

---

## Infrastructure

| Property | Value |
|---|---|
| Cloud platform | Microsoft Azure |
| VM type | Standard_D8s_v3 |
| CPU | 8 vCPU |
| RAM | 64 GB |
| Storage | 4× NVMe SSD combined via LVM = 1.8 TB at /data |
| OS | Ubuntu 24.04 LTS |
| Java | OpenJDK 11 |
| Apache Spark | 3.5.1 with Hadoop 3 |
| Python | 3.12.3 |

---

## Pipeline Versions

| Version | Description | Status |
|---|---|---|
| v1–v6 | Schema validation, error handling, multi-label rules, cast fixes | Development |
| v7 | toDF() rename + numeric casts inside when() — fixes zero-suspect bug | Fixed |
| v8 | RULES moved after Spark start, upper() on protocol — fixes AssertionError | Fixed |
| **v9** | 6 thesis additions. Pure random sampling → 4,999 background, 1 attack | **Diagnostic** |
| **v10** | Stratified sampling. All v9 additions retained. 4,700 + 290 + 4 = 4,994 records | **Production** |

> **All experimental results are based on v10.** v9 is retained as documentation of the scientific problem discovery process.

---

## Step 1 — Environment Setup

```bash
# Connect to VM
ssh thamer@74.162.89.167

# Install Java
sudo apt update && sudo apt install openjdk-11-jdk -y
java -version

# Download and install Apache Spark 3.5.1
cd /opt && sudo wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
sudo tar -xvzf spark-3.5.1-bin-hadoop3.tgz
sudo mv spark-3.5.1-bin-hadoop3 spark

# Test Spark
/opt/spark/bin/pyspark

# Install AzCopy
wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
tar -xvf azcopy.tar.gz
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
sudo chmod +x /usr/bin/azcopy
azcopy --version
```

---

## Step 2 — Storage Setup (LVM — 1.8 TB)

```bash
# Combine 4 NVMe SSDs into one logical volume
sudo pvcreate /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo vgcreate vg_data /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo lvcreate -l 100%FREE -n lv_data vg_data

# Format and mount
sudo mkfs.xfs /dev/vg_data/lv_data
sudo mkdir -p /data
sudo mount /dev/vg_data/lv_data /data
sudo chown -R thamer:thamer /data
sudo chmod 775 /data

# Make mount persistent across reboots
sudo blkid /dev/vg_data/lv_data
echo 'UUID=<your-uuid> /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a

# Verify — should show 1.8 TB
df -h /data
```

---

## Step 3 — Dataset Acquisition

The UGR'16 dataset was downloaded via **Azure Data Factory V2** because the UGR'16 server uses HTTP redirects that prevent direct wget/curl access.

```bash
# Transfer from Azure Blob Storage to VM using AzCopy
azcopy copy "<SAS_URL>" "/data/august_week4_csv.tar.gz"

# Extract in background
nohup tar -xvzf /data/august_week4_csv.tar.gz -C /data/ > extract.log 2>&1 &
tail -f extract.log

# Verify extraction
ls -lh /data
du -sh /data/august.week4.csv.uniqblacklistremoved   # → ~77.6 GB

# Remove compressed file to free space
rm -f /data/august_week4_csv.tar.gz
```

---

## Step 4 — CSV to Parquet Conversion

```bash
# Run conversion script
/opt/spark/bin/spark-submit /data/convert_to_parquet.py

# Verify output
du -sh /data/parquet              # → ~14 GB
find /data/parquet -name "*.parquet" | wc -l   # → 311 files
ls /data/parquet/_SUCCESS          # → confirms completion

# Fix nested directory structure if needed
mv /data/parquet/august_week4/august_week4/* /data/parquet/
rmdir /data/parquet/august_week4/august_week4
```

**Why Parquet?** Parquet is a columnar binary format that provides 5.5× compression over CSV, preserves schema, and enables significantly faster Spark read performance through partition-level parallel processing.

---

## Step 5 — Inspect Schema

```bash
/opt/spark/bin/pyspark --driver-memory 10g --executor-memory 42g
```

```python
df_raw = spark.read.parquet("/data/parquet")
print("Column count:", len(df_raw.columns))   # → 13
print("Column names:", df_raw.columns)         # → ['_c0','_c1',...,'_c12']
for name, dtype in df_raw.dtypes:
    print(f"{name} -> {dtype}")               # ALL → string
df_raw.show(3, truncate=False)
```

**Confirmed schema — all 13 columns are string type:**

| Column | Semantic Name | Cast To | Notes |
|---|---|---|---|
| _c0 | timestamp | string | YYYY-MM-DD HH:MM:SS |
| _c1 | duration | double | Flow duration seconds |
| _c2 | src_ip | string | Source IP |
| _c3 | dst_ip | string | Destination IP |
| _c4 | src_port | int | Source port |
| _c5 | dst_port | int | Destination port |
| _c6 | protocol | string | TCP/UDP/ICMP/IPv6... |
| _c7 | flags | string | TCP flags |
| _c8 | fwd_pkts | int | Forward packets |
| _c9 | bwd_pkts | int | Backward packets |
| _c10 | total_pkts | long | Total packets (long not int — overflow risk) |
| _c11 | total_bytes | long | Total bytes (long not int — overflow risk) |
| _c12 | label | string | background/blacklist/anomaly-spam/anomaly-sshscan |

---

## Step 6 — Run the Pipeline (v10)

```bash
/opt/spark/bin/spark-submit \
  --driver-memory 10g \
  --executor-memory 42g \
  /data/ugr16_pipeline_v10.py
```

**Runtime:** 5.25 minutes for 862,526,861 records on 8-CPU Azure VM.

### What v10 does — step by step

| Step | Action | Result |
|---|---|---|
| 1 | Load Parquet, validate 13 columns, rename via toDF() | 862,526,861 records loaded |
| 2a | Full dataset label distribution | Baseline class balance documented |
| 2b | Rule contribution count — 1 scan, 4 rules | 108,139,105 suspect records identified |
| 2c | Suspect selection via concat_ws() | Multi-label tagging per record |
| 2d | Suspect pool label distribution | 99.98% background confirmed |
| 2e | Rule overlap distribution | 9 rule combinations documented |
| 3 | SHA-256 record_id — null-safe | Unique stable ID per record |
| 4 | **Stratified sampling** — key change from v9 | 4,700 + 290 + 4 = 4,994 records |
| 5 | Nested sub-samples 1000 ⊂ 3000 ⊂ 5000 | Three comparable sample sizes |
| 6 | Timestamp range | 2016-08-22 to 2016-08-29 |
| 7 | Build API and eval column sets | Label excluded from API files |
| 8 | Write 6 JSONL files | All line counts validated |
| 9 | Integrity checks | record_id uniqueness PASSED |
| 10 | Processing report JSON + TXT | Full statistics documented |

### Why v9 → v10 (stratified sampling)

v9 used pure random sampling from the suspect pool:

```
Suspect pool: 99.98% background, 0.02% blacklist, 0.00% sshscan
Random draw of 5,000 → 4,999 background, 1 blacklist, 0 sshscan
```

With 1 attack record, Precision/Recall/F1 cannot be computed. v10 fixes this:

```python
# Sample each class independently
bg_sample  = bg_df.orderBy(rand(seed=42)).limit(4700)   # background
bl_sample  = bl_df.orderBy(rand(seed=42)).limit(290)    # blacklist
ssh_sample = ssh_df                                       # all 4 sshscan
sample_5000 = bg_sample.union(bl_sample).union(ssh_sample)
```

### v10 sample composition

| Class | Count | % | Evaluation |
|---|---|---|---|
| background | 4,700 | 94.11% | Valid baseline |
| blacklist | 290 | 5.81% | Valid — sufficient for metrics |
| anomaly-sshscan | 4 | 0.08% | Observation only |
| anomaly-spam | 0 | 0% | Absent from suspect pool |
| **TOTAL** | **4,994** | **100%** | **Valid** |

> The sample is **intentionally imbalanced**, not equally balanced. 94% background reflects real network conditions. Equal balancing would produce unrealistically optimistic LLM performance metrics.

### Nested sample label distributions

| Sample | Total | background | blacklist | sshscan | CI (95%) |
|---|---|---|---|---|---|
| sample_1000 | 1,000 | 927 (92.7%) | 73 (7.3%) | 0 | ±3.10% |
| sample_3000 | 3,000 | 2,812 (93.7%) | 185 (6.2%) | 3 (0.1%) | ±1.79% |
| sample_5000 | 4,994 | 4,700 (94.1%) | 290 (5.8%) | 4 (0.08%) | ±1.39% |

---

## Step 7 — Run the LLM Evaluation (v2)

### Set API keys as environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AI..."
export DEEPSEEK_API_KEY="sk-..."
```

### Run evaluation on sample_1000 (recommended first)

```bash
python3 /data/ugr16_llm_eval_v2.py \
  --sample-dir /data/llm_samples_v10 \
  --output-dir /data/llm_eval_results_v2 \
  --sample-size 1000 \
  --models chatgpt gemini deepseek claude
```

### Key features of eval v2

| Feature | Description |
|---|---|
| No hardcoded keys | All API keys read from environment variables |
| Resume mode | Crashed at record 500? Restart — skips already processed records |
| Retry + backoff | 5 retries with exponential backoff on rate limits and server errors |
| Structured parsing | Expects JSON response — safer than substring matching |
| Incremental writes | Predictions saved after every record — no data lost on crash |
| 7 metrics | Precision, Recall, Specificity, F1, FPR, FNR, Accuracy |

### Prompt used (identical for all 4 models)

```
You are a network security analyst specialising in intrusion detection.

Classify the following network flow record as either attack or normal.

Network flow JSON:
{record_json}

Field meanings: ...

Required JSON format:
{"label":"attack","confidence":0.0,"reason":"short reason"}
```

### Label mapping

```python
background      → normal
blacklist       → attack
anomaly-sshscan → attack
```

### Metrics computed

| Metric | Formula | Primary? |
|---|---|---|
| Precision | TP / (TP + FP) | ✓ Primary |
| Recall / Sensitivity | TP / (TP + FN) | ✓ Primary |
| F1-score | 2×P×R / (P+R) | ✓ Primary |
| False Positive Rate | FP / (FP + TN) | ✓ Primary |
| Specificity | TN / (TN + FP) | Secondary |
| False Negative Rate | FN / (FN + TP) | Secondary |
| Accuracy | (TP+TN) / total | Reference only |

> Accuracy is not the primary metric. With 94% background, a model predicting "normal" for everything achieves 94% accuracy while detecting zero attacks.

---

## Output Files

### Pipeline output (v10)
```
/data/llm_samples_v10/
├── sample_1000_api.jsonl    # 1,000 lines — no label
├── sample_3000_api.jsonl    # 3,000 lines — no label
├── sample_5000_api.jsonl    # 4,994 lines — no label
├── sample_1000_eval.jsonl   # 1,000 lines — with label
├── sample_3000_eval.jsonl   # 3,000 lines — with label
├── sample_5000_eval.jsonl   # 4,994 lines — with label
├── processing_report.json   # Full statistics
└── processing_report.txt    # Human-readable report
```

### Evaluation output (v2)
```
/data/llm_eval_results_v2/
├── predictions_1000_claude.jsonl
├── predictions_1000_chatgpt.jsonl
├── predictions_1000_gemini.jsonl
├── predictions_1000_deepseek.jsonl
├── eval_report_1000.json
├── eval_report_1000.txt
└── errors_1000.jsonl
```

---

## Dataset Statistics

### Full dataset label distribution

| Label | Count | % |
|---|---|---|
| background | 854,170,414 | 99.03% |
| anomaly-spam | 5,287,316 | 0.61% |
| blacklist | 3,069,118 | 0.36% |
| anomaly-sshscan | 12 | 0.00% |

### Suspect selection rule contributions

| Rule | Condition | Matched | % of Full |
|---|---|---|---|
| sensitive_port | dst_port in {22,23,445,3389} | 95,972,228 | 11.13% |
| icmp_traffic | upper(protocol) == 'ICMP' | 10,606,661 | 1.23% |
| high_packet_count | total_pkts > 1,000 | 1,376,391 | 0.16% |
| high_bytes | total_bytes > 1,000,000 | 867,298 | 0.10% |

---

## Models Evaluated

| Model | Provider | Version | API |
|---|---|---|---|
| Claude | Anthropic | claude-3-haiku-20240307 | api.anthropic.com |
| ChatGPT | OpenAI | gpt-4o-mini | api.openai.com |
| Gemini | Google | gemini-1.5-flash-latest | generativelanguage.googleapis.com |
| DeepSeek | DeepSeek | deepseek-chat | api.deepseek.com |

> All models are used as **pretrained APIs**. No fine-tuning or training was performed.

---

## Declared Limitations

1. **Temporal stratification:** Random sampling without time-window stratification. Sample spans 2016-08-22 to 2016-08-29. Declared as threat to external validity.
2. **anomaly-spam absent:** Spam flows do not trigger any of the four suspect rules. Spam detection is excluded from evaluation scope.
3. **anomaly-sshscan:** Only 4 records in suspect pool. Statistical conclusions not possible for this class.
4. **Threshold selection:** Thresholds (1,000 pkts, 1 MB) chosen as design decisions. Sensitivity analysis is future work.
5. **Rule dominance:** sensitive_port captures 88.74% of suspect records. Four-rule design operates effectively as single-rule.
6. **Minority class oversampling:** blacklist and sshscan are overrepresented relative to natural prevalence. Results reflect evaluation performance, not real-world detection rates.

---

## Reference

Maciá-Fernández, G., Camacho, J., Magán-Carrión, R., García-Teodoro, P., & Therón, R. (2018). UGR'16: A new dataset for the evaluation of cyclostationarity-based network IDSs. *Computers & Security*, 73, 411–424. https://doi.org/10.1016/j.cose.2017.11.004

---

## Contact

**Thamer Aldraiwa**
PhD Candidate — University of Granada
Email: thamer@correo.ugr.es
