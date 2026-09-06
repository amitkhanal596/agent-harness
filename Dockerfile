# syntax=docker/dockerfile:1

# --- build stage: install dependencies into a virtualenv -----------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the install needs first, so dependency layers cache across code edits.
# `migrations/` is force-included into the wheel as package data, so it must be present.
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
RUN pip install --upgrade pip && pip install .

# --- runtime stage -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# The agent shells out to allowlisted read-only tools; git is one of them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Never run the agent as root: it executes model-chosen commands, so a non-root user is
# the last line of defence behind the run_shell allowlist.
RUN useradd --create-home --uid 10001 agent

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=agent:agent migrations ./migrations
COPY --chown=agent:agent evals ./evals

USER agent

ENTRYPOINT ["agent"]
CMD ["--help"]
