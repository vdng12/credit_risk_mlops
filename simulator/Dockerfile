# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# Data generator image - the `simulator` workspace member (src/simulator layout).
#
# The build context is the WORKSPACE ROOT (requires uv.lock + pyproject.toml):
#
#   docker build -f simulator/Dockerfile -t risk-simulator:latest .
#   k3d image import risk-simulator:latest --name risk-cluster
#
# Runtime: the `simulator` console script (project.scripts in
# simulator/pyproject.toml) -> .venv/bin/simulator. All sink extras (S3 /
# Postgres / Kafka) are compiled into the image via `--all-extras`.
#
# On BuildKit builds you can add `--mount=type=cache,target=/root/.cache/uv`
# to the two RUN steps to persist the uv package cache across builds.
# -----------------------------------------------------------------------------

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Layer 1 - resolve + fetch the *entire* dependency closure from the lockfile.
# Only invalidated when uv.lock or a pyproject.toml changes.
COPY uv.lock pyproject.toml ./
COPY simulator/pyproject.toml ./simulator/
RUN uv sync --frozen --no-dev --no-install-workspace

# Layer 2 - install the simulator package itself + its runtime extras.
COPY simulator/src ./simulator/src
RUN uv sync --frozen --no-dev --package simulator --all-extras

# -----------------------------------------------------------------------------
# Lean runtime stage: only the frozen venv + Python, no uv/build toolchain.
# -----------------------------------------------------------------------------
FROM python:3.13-slim

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /data
COPY --from=build /app /app

ENTRYPOINT ["simulator"]
CMD ["--help"]