# MLOps Pipeline for a Digit Recognition Service

An end-to-end, **CPU-only** MLOps pipeline for handwritten-digit recognition. It
showcases the engineering discipline that turns a notebook model into an
operable service: config-driven training, MLflow experiment tracking, an
**under/overfitting** study, a **latency benchmark**, a FastAPI serving layer
with health and Prometheus endpoints, containerisation, and CI.

Its **original contribution** is an automated **model-selection + drift-alert
module**: instead of a human hand-picking the served model, a selector promotes
a champion by a generalisation-aware rule, and a monitor raises an alert the
moment live accuracy drops below baseline. A reproducible experiment proves the
module's value against the manual baseline.

> Student: Jhanvi Titoria · BITS ID 2024da04367 · M.Tech. Data Science & Engineering

## Dissertation question

> **Can an automated selection-and-monitoring module choose better-generalising
> models and catch performance degradation faster than manual selection?**

The pipeline is run two ways on identical data and candidate models -
**without** the module (a human hard-picks `balanced`, no monitor) and **with**
it (auto-promoted champion + drift monitor) - and the difference is reported in
a single proof table (accuracy, generalisation gap, selection effort, drift
detection, errors caught). See [Proof of value](#proof-of-value).

## Highlights

- **Config-driven dataset switch**: the same pipeline runs on the bundled
  scikit-learn **8×8 digits** (1,797 images, offline) *or* full **MNIST**
  (70,000 × 784, fetched once via OpenML), with optional stratified subsampling
  to keep kernel methods tractable at scale.
- **Automated champion selection** (`selector.py`): ranks every run and promotes
  `models/champion.joblib` by a transparent rule — *highest validation accuracy
  subject to a train/val gap ≤ 0.08* — so the served model is chosen, not
  guessed.
- **Drift / performance-drop monitor** (`monitor.py` + `/monitor` endpoint):
  scores the champion on labelled batches, exposes `digit_model_accuracy` and
  `digit_drift_alerts_total` to Prometheus, and alerts when accuracy falls below
  baseline.
- **Proof experiment** (`experiment.py`): manual vs automated, side by side, in
  a regenerated table + figures.
- **Three contrasted models** trained on identical stratified splits to
  illustrate the bias–variance trade-off:
  | Run | Model | Train | Val | Test | Train–Val gap | Verdict |
  |-----|-------|------:|----:|-----:|--------------:|---------|
  | underfit | shallow decision tree (depth 2) | 32.3% | 31.6% | 30.3% | +0.7 pp | underfit |
  | balanced | RBF-kernel SVM | 100% | 99.0% | 98.3% | +1.0 pp | good fit |
  | overfit  | fully grown decision tree | 100% | 83.0% | 78.6% | +17.0 pp | overfit |
- **MLflow tracking** of params, metrics, the train/val gap and a
  `fitting_verdict` tag, plus logged model artifacts.
- **Latency benchmark** (single-sample p50/p95/p99 + batch throughput) with a
  Locust load test for the HTTP path.
- **FastAPI service** exposing `/predict`, `/livez`, `/readyz`, `/metrics`.
- **Containerisation** (separate train/serve Dockerfiles) + **GitHub Actions CI**.
- **Abstract report** generated to `report/2024da04367-digit-recognition-abstract.docx`.

## Project layout

```
digit-rec/
├── configs/                 # underfit/balanced/overfit + logreg/rf/knn/mlp + mnist_*
├── src/mlops_digit/
│   ├── data.py              # dataset switch (digits/MNIST) + stratified splits
│   ├── models.py            # model factory (scaler + estimator pipeline)
│   ├── train.py             # config-driven training + MLflow logging
│   ├── analyze.py           # trains the 3 contrasted runs + fitting plots
│   ├── selector.py          # CONTRIBUTION: auto-selects & promotes champion
│   ├── monitor.py           # CONTRIBUTION: drift / performance-drop alerting
│   ├── experiment.py        # manual-vs-automated proof harness + tables
│   ├── serve.py             # FastAPI service (/predict, /monitor, /metrics, health)
│   └── benchmark.py         # inference latency benchmark
├── models/                  # champion.joblib + champion.json (promoted model)
├── load/locustfile.py       # HTTP load test
├── tests/                   # pytest suite (data, models, serving, selector, monitor)
├── report/build_report.py   # generates the abstract .docx from real results
├── Dockerfile.train / Dockerfile.serve
└── .github/workflows/ci.yml
```

## Quickstart (Windows / PowerShell)

```powershell
python -m venv .venv
# behind a corporate proxy add: --trusted-host pypi.org --trusted-host files.pythonhosted.org
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

### 1. Train the three models + under/overfitting analysis
```powershell
.\.venv\Scripts\python.exe -m mlops_digit.analyze
```
Writes models to `models/`, metrics to `artifacts/`, and figures to
`reports/figures/`. Train a single run with
`python -m mlops_digit.train --config configs/balanced.yaml`.

### 2. Inspect experiments in MLflow
```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri ./mlruns
```

### 3. Latency benchmark
```powershell
.\.venv\Scripts\python.exe -m mlops_digit.benchmark --iterations 800
```

### 4. Serve the model
```powershell
# No MODEL_PATH needed: the service loads models/champion.joblib if present,
# else falls back to models/balanced.joblib. Override with MODEL_PATH if desired.
.\.venv\Scripts\python.exe -m mlops_digit.serve
# POST http://localhost:8000/predict  body: {"instances": [[<64 floats>]]}
# POST http://localhost:8000/monitor  body: {"instances": [[...]], "labels": [..]}
```

### 5. Load test (optional)
```powershell
.\.venv\Scripts\locust.exe -f load/locustfile.py --host http://localhost:8000
```

### 6. Tests
```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 7. Regenerate the abstract report
```powershell
.\.venv\Scripts\python.exe report/build_report.py
```

## Automated selection, monitoring & proof (contribution)

The contribution is three small modules that close the loop from "many trained
runs" to "one operable, watched champion".

### Auto-select & promote a champion
```powershell
# Ranks every run in artifacts/metrics/ and writes models/champion.joblib + champion.json
.\.venv\Scripts\python.exe -m mlops_digit.selector --max-gap 0.08 --metric val_accuracy
```
The rule is *highest validation accuracy subject to a train/val gap ≤ `max-gap`*
(ties broken on test accuracy); over-fit runs are excluded. `champion.json`
records the chosen run and its **baseline accuracy** — the reference the monitor
watches.

### Monitor for performance drops
```powershell
# Score the champion on clean test data (exit 0 = healthy)
.\.venv\Scripts\python.exe -m mlops_digit.monitor --threshold 0.05
# Simulate input drift and prove the alert fires (exit 1 = alert)
.\.venv\Scripts\python.exe -m mlops_digit.monitor --corrupt 0.6 --threshold 0.05
```
The non-zero exit code on alert is the hook for CI / Alertmanager. Live
monitoring is also exposed by the service: `POST /monitor` updates the
`digit_model_accuracy` gauge and increments `digit_drift_alerts_total` on
`/metrics`.

### Proof of value

Run the manual-vs-automated experiment end to end:
```powershell
.\.venv\Scripts\python.exe -m mlops_digit.experiment              # digits (offline)
.\.venv\Scripts\python.exe -m mlops_digit.experiment --dataset mnist --subsample 12000
```
It trains all candidates, lets the selector promote a champion, induces input
drift across severities, and writes `artifacts/experiment_results.json`,
`reports/proof_table.md`, `reports/figures/proof_table.png` and
`drift_sweep.png`. Latest digits result:

| Metric | WITHOUT module (manual) | WITH module (automated) |
|---|---|---|
| Model served | balanced (svm) — hand-picked | balanced (svm) — auto-selected |
| Test accuracy | 0.9833 | 0.9833 (+0.0000) |
| Train–val gap | +0.0104 | +0.0104 |
| Selection effort | manual review of all runs | ~0.05 ms (automatic) |
| Drift detection | none (manual / reactive) | alerts at severity 0.2 |
| Errors caught before users | 0 (no monitor) | 311 flagged at first alert |

On the tiny digits set the automated rule happens to agree with the manual pick
(SVM is genuinely best), so the value shows up as **free, reproducible selection
and early drift detection**. Re-run with `--dataset mnist` to test whether the
"best" model — and hence the gap between manual and automated — changes at scale.

## Docker

```bash
docker build -f Dockerfile.train -t digit-rec-train .
docker run --rm -v "$PWD/models:/app/models" digit-rec-train

docker build -f Dockerfile.serve -t digit-rec-serve .
docker run --rm -p 8000:8000 digit-rec-serve
```

## Configuration

Each YAML config selects a `model.type` (any of `logistic_regression`, `svm`,
`mlp`, `decision_tree`, `random_forest`, `knn`) and its `params`. Training
auto-labels each run `underfit`, `good_fit`, or `overfit` from the train/val
accuracy gap.

The `data` block also selects the dataset and (optionally) a stratified
subsample:

```yaml
data:
  dataset: mnist      # "digits" (default, offline) or "mnist" (70k, via OpenML)
  subsample: 12000    # optional: stratified subset before splitting
  test_size: 0.2
  val_size: 0.2
  random_state: 42
```

MNIST is downloaded once and cached under `data/`; later runs are offline.
`mnist_logreg.yaml` (full set) and `mnist_svm.yaml` (subsampled) are ready-made
examples.

## Further reading

The contribution sits at the intersection of a few well-studied areas — useful
anchors for the dissertation's literature review:

- **Continuous Delivery for ML (CD4ML)** — promotion/rollback of models as a
  delivery pipeline (Sato, Wider & Windheuser, ThoughtWorks).
- **MLflow Model Registry** — stage/alias-based promotion of a "champion" model.
- **Data & concept drift detection** — ADWIN, Page–Hinkley, Kolmogorov–Smirnov
  tests; tooling such as Evidently AI and NannyML. Gama et al. (2014), *A Survey
  on Concept Drift Adaptation*.
- **AutoML / model selection** — auto-sklearn, TPOT; positions this project's
  rule-based selector as a lightweight, interpretable alternative.
- **ML test/production readiness** — Breck et al. (2017), *The ML Test Score*;
  Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.
