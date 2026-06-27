"""MLOps pipeline for a digit recognition service.

A compact, CPU-only, end-to-end MLOps demo built around the scikit-learn
8x8 handwritten-digits dataset. It covers config-driven training of three
contrasted models (underfit / balanced / overfit), MLflow experiment
tracking, under/overfitting analysis, a FastAPI serving layer, and a
latency benchmark.
"""

__version__ = "1.0.0"
