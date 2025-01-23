# Base image
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src/rice_images src/rice_images
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY configs/ configs/

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir


ENTRYPOINT ["python", "-u", "src/rice_images/evaluate.py"]
