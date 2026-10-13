import pytest
from src.personal_baseline_engine import BaselineFormulas

def test_absolute_deviation():
    assert BaselineFormulas.absolute_deviation(10, 5) == 5
    assert BaselineFormulas.absolute_deviation(5, 10) == -5

def test_relative_deviation():
    assert BaselineFormulas.relative_deviation(10, 5) == 1.0
    assert BaselineFormulas.relative_deviation(5, 10) == -0.5
    assert BaselineFormulas.relative_deviation(5, 0) == 0.0

def test_z_score():
    assert BaselineFormulas.z_score(10, 5, 2.5) == 2.0
    assert BaselineFormulas.z_score(10, 5, 0.0) == 5.0 # Low variance defaults to 1.0

def test_robust_z_score():
    assert BaselineFormulas.robust_z_score(10, 5, 2.0) == 1.68625
    assert BaselineFormulas.robust_z_score(10, 5, 0.0) == 3.3725 # Low MAD defaults to 1.0
