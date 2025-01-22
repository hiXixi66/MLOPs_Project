# Change from latest to a specific version if your requirements.txt
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY src/ /src/
COPY requirements_backend.txt /requirements_backend.txt
COPY requirements.txt /requirements.txt
COPY models/ /models/
COPY configs/ /configs/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements_backend.txt --no-cache-dir --verbose
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE $PORT
CMD exec uvicorn --port $PORT --host 0.0.0.0 src.rice_images.backend:app
