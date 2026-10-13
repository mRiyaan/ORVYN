import pytest
import pandas as pd
import numpy as np
from src.input_validation import CSVValidator

@pytest.fixture
def validator():
    return CSVValidator()

def test_valid_file(validator):
    df = pd.DataFrame({
        "timestamp": np.arange(0, 100, 1/64),
        "wrist_BVP": np.random.randn(6400),
        "wrist_EDA": np.random.randn(6400),
        "wrist_TEMP": np.random.randn(6400)
    })
    result = validator.validate(df)
    assert result["valid"] == True

def test_missing_required_signal(validator):
    df = pd.DataFrame({
        "timestamp": np.arange(0, 100, 1/64),
        "wrist_BVP": np.random.randn(6400)
    })
    result = validator.validate(df)
    assert result["valid"] == False
    assert "Missing required channels" in result["errors"]

def test_invalid_numeric(validator):
    df = pd.DataFrame({
        "timestamp": np.arange(0, 100, 1/64),
        "wrist_BVP": ["a"] * 6400,
        "wrist_EDA": np.random.randn(6400),
        "wrist_TEMP": np.random.randn(6400)
    })
    result = validator.validate(df)
    assert result["valid"] == False
    assert "Invalid numeric format for wrist_BVP" in result["errors"]

def test_invalid_timestamps(validator):
    df = pd.DataFrame({
        "timestamp": ["a"] * 6400,
        "wrist_BVP": np.random.randn(6400),
        "wrist_EDA": np.random.randn(6400),
        "wrist_TEMP": np.random.randn(6400)
    })
    result = validator.validate(df)
    assert result["valid"] == False
    assert "Invalid timestamp format" in result["errors"]

def test_duplicate_timestamps(validator):
    df = pd.DataFrame({
        "timestamp": np.zeros(6400),
        "wrist_BVP": np.random.randn(6400),
        "wrist_EDA": np.random.randn(6400),
        "wrist_TEMP": np.random.randn(6400)
    })
    result = validator.validate(df)
    assert result["valid"] == False
    assert "Duplicate timestamps found" in result["errors"]

def test_large_gap(validator):
    ts = np.arange(0, 100, 1/64)
    ts[3200:] += 5.0 # create a 5-second gap
    df = pd.DataFrame({
        "timestamp": ts,
        "wrist_BVP": np.random.randn(6400),
        "wrist_EDA": np.random.randn(6400),
        "wrist_TEMP": np.random.randn(6400)
    })
    result = validator.validate(df)
    assert result["valid"] == False
    assert "Large gap in timestamps" in result["errors"]

def test_too_short_recording(validator):
    df = pd.DataFrame({
        "timestamp": np.arange(0, 30, 1/64), # only 30 seconds
        "wrist_BVP": np.random.randn(1920),
        "wrist_EDA": np.random.randn(1920),
        "wrist_TEMP": np.random.randn(1920)
    })
    result = validator.validate(df)
    assert result["valid"] == False
    assert "Recording too short" in result["errors"]
