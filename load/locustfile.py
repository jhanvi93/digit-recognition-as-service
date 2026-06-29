"""Locust load test for the digit-recognition service.

Run the service first, then:
    locust -f load/locustfile.py --host http://localhost:8000

Or headless for a fixed duration:
    locust -f load/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 60s
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

# A representative 64-feature digit vector (values in the 0-16 range used by
# the scikit-learn digits dataset). Slightly perturbed per request.
_BASE_VECTOR = [
    0, 0, 5, 13, 9, 1, 0, 0,
    0, 0, 13, 15, 10, 15, 5, 0,
    0, 3, 15, 2, 0, 11, 8, 0,
    0, 4, 12, 0, 0, 8, 8, 0,
    0, 5, 8, 0, 0, 9, 8, 0,
    0, 4, 11, 0, 1, 12, 7, 0,
    0, 2, 14, 5, 10, 12, 0, 0,
    0, 0, 6, 13, 10, 0, 0, 0,
]


class DigitUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self) -> None:
        vector = [max(0.0, v + random.uniform(-1.0, 1.0)) for v in _BASE_VECTOR]
        self.client.post("/predict", json={"instances": [vector]})

    @task(1)
    def health(self) -> None:
        self.client.get("/readyz")
