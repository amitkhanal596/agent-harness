"""A tiny calculator library. `apply_discount` carries the bug the eval hunts for."""


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def apply_discount(price: float, percent: float) -> float:
    """Return `price` reduced by `percent`.

    BUG: divides by 10 instead of 100, so every discount is 10x too large.
    """
    return price - (price * percent / 10)


def total(prices: list[float]) -> float:
    return sum(prices)
