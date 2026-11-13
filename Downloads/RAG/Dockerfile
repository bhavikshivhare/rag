FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# System deps kept minimal: faiss-cpu ships manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8000

# Default: run the baseline experiment and write the plain-text report into /app/reports.
CMD ["python", "-m", "src.evaluation.run_evaluation", "--config", "configs/experiments/exp_001_baseline.yaml"]
