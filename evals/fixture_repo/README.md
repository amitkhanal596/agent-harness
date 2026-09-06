# Fixture repo

A deliberately broken mini-project used by the eval set. `src/calculator.py`'s
`apply_discount` divides by 10 instead of 100, so `tests/test_calculator.py::test_apply_discount`
fails in CI. `ci_log.txt` is the captured failure output.
