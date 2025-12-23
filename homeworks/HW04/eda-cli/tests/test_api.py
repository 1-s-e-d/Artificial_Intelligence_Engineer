"""Тесты для HTTP API эндпоинтов."""
from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from eda_cli.api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Тесты для эндпоинта /health."""

    def test_health_returns_ok(self):
        """Проверка, что health-check возвращает статус ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "dataset-quality"
        assert data["version"] == "0.3.0"


class TestMetricsEndpoint:
    """Тесты для эндпоинта /metrics."""

    def test_metrics_returns_stats(self):
        """Проверка, что /metrics возвращает статистику."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "endpoint_calls" in data
        assert "errors" in data


class TestQualityEndpoint:
    """Тесты для эндпоинта /quality."""

    def test_quality_with_valid_data(self):
        """Проверка корректной работы с валидными данными."""
        payload = {
            "n_rows": 1000,
            "n_cols": 10,
            "max_missing_share": 0.15,
            "numeric_cols": 6,
            "categorical_cols": 4
        }
        response = client.post("/quality", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "ok_for_model" in data
        assert "quality_score" in data
        assert "message" in data
        assert "latency_ms" in data
        assert "flags" in data
        assert "dataset_shape" in data

    def test_quality_with_poor_data(self):
        """Проверка работы с данными низкого качества."""
        payload = {
            "n_rows": 100,
            "n_cols": 200,
            "max_missing_share": 0.8,
            "numeric_cols": 0,
            "categorical_cols": 200
        }
        response = client.post("/quality", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ok_for_model"] == False
        assert data["quality_score"] < 0.7

    def test_quality_with_invalid_data(self):
        """Проверка валидации при некорректных данных."""
        payload = {
            "n_rows": -1,
            "n_cols": 10,
            "max_missing_share": 0.15,
            "numeric_cols": 6,
            "categorical_cols": 4
        }
        response = client.post("/quality", json=payload)
        assert response.status_code == 422


class TestQualityFromCsvEndpoint:
    """Тесты для эндпоинта /quality-from-csv."""

    def test_quality_from_csv_valid(self):
        """Проверка работы с валидным CSV."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "z", "x", "y"],
            "c": [10.5, 20.3, 30.1, 40.0, 50.2]
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
        response = client.post("/quality-from-csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "ok_for_model" in data
        assert "quality_score" in data
        assert data["dataset_shape"]["n_rows"] == 5
        assert data["dataset_shape"]["n_cols"] == 3

    def test_quality_from_csv_empty(self):
        """Проверка обработки пустого CSV."""
        csv_bytes = "a,b,c\n".encode("utf-8")

        files = {"file": ("empty.csv", io.BytesIO(csv_bytes), "text/csv")}
        response = client.post("/quality-from-csv", files=files)

        assert response.status_code == 400

    def test_quality_from_csv_with_missing(self):
        """Проверка работы с CSV, содержащим пропуски."""
        df = pd.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": ["x", "y", None, "x", None],
            "c": [10.5, 20.3, 30.1, None, None]
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        files = {"file": ("missing.csv", io.BytesIO(csv_bytes), "text/csv")}
        response = client.post("/quality-from-csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["flags"]["has_high_missing"] in [True, False]


class TestQualityFlagsFromCsvEndpoint:
    """Тесты для эндпоинта /quality-flags-from-csv."""

    def test_quality_flags_from_csv_valid(self):
        """Проверка получения полных флагов качества."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "constant": ["same", "same", "same", "same", "same"],
            "normal": ["a", "b", "c", "d", "e"],
            "zeros": [0, 0, 0, 0, 1]
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        files = {"file": ("flags_test.csv", io.BytesIO(csv_bytes), "text/csv")}
        response = client.post("/quality-flags-from-csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "flags" in data
        assert "quality_score" in data
        assert "latency_ms" in data
        assert "dataset_shape" in data
        assert data["dataset_shape"]["n_rows"] == 5
        assert data["dataset_shape"]["n_cols"] == 4
        assert data["flags"]["has_constant_columns"] == True

    def test_quality_flags_score_range(self):
        """Проверка диапазона quality_score."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        files = {"file": ("score_test.csv", io.BytesIO(csv_bytes), "text/csv")}
        response = client.post("/quality-flags-from-csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["quality_score"] <= 100
