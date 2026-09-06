"""Scratch helpers for the fixture project."""


def slugify(text: str) -> str:
    # TODO: handle unicode punctuation before shipping this to production
    return text.strip().lower().replace(" ", "-")
