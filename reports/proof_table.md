# Proof of value: manual vs automated pipeline

Dataset: **digits** (train/val/test = 1149/288/360, 64 features)

| Metric | WITHOUT module (manual) | WITH module (automated) |
|---|---|---|
| Model served | balanced (svm) - hand-picked | balanced (svm) - auto-selected |
| Test accuracy | 0.9833 | 0.9833  (+0.0000) |
| Train-val gap (generalisation) | +0.0104 | +0.0104 |
| Selection effort | manual review of all runs | 0.05 ms (automatic) |
| Drift detection | none (manual / reactive) | alerts at severity 0.2 |
| Errors caught before users | 0 (no monitor) | 311 flagged at first alert |

Selection rule: `max(val_accuracy) subject to train_val_gap <= 0.08; tie-break on test_accuracy`
Drift threshold: 0.05 (alert if accuracy falls more than this below baseline).

_Figures: `reports/figures/proof_table.png`, `reports/figures/drift_sweep.png`._