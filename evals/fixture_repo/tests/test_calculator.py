"""The CI suite for the fixture repo. `test_apply_discount` fails on purpose."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calculator import add, apply_discount, subtract, total


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(9, 4) == 5


def test_apply_discount():
    # 10% off 100 should be 90.
    assert apply_discount(100, 10) == 90


def test_total():
    assert total([1.5, 2.5]) == 4.0
