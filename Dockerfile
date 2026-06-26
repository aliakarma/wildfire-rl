# Reproducible CPU runtime for training / evaluation / artifact review.
# (For GPU, swap the base for an official pytorch/pytorch CUDA image.)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WILDFIRE_REPO_ROOT=/app

WORKDIR /app

# System deps occasionally needed by scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install the package itself.
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install -e .

# Large artifacts (data/, models/) are mounted at runtime, not baked in:
#   docker run --rm -v $PWD/data:/app/data -v $PWD/models:/app/models \
#       wildfire-rl wildfire-rl transfer --config configs/experiment/transfer.yaml
ENTRYPOINT ["wildfire-rl"]
CMD ["info"]
