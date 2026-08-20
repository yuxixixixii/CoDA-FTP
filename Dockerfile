FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /artifact

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/FlakeFlaggerFeaturesTypes.csv data/FlakeFlaggerFeaturesTypes.csv
RUN mkdir -p data outputs/coda_ftp_primary

# Mount the downloaded fused CSV at /artifact/data when running the image.
ENTRYPOINT ["bash", "scripts/run_main.sh"]
