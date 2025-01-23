FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ /src/
COPY requirements_frontend.txt /requirements_frontend.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt


ENTRYPOINT ["streamlit", "run", "src/rice_images/frontend.py", "--server.port", "8001", "--server.address=0.0.0.0"]
