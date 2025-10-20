# pin to stable so apt packages exist
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Only the minimal GUI libs some wheels expect
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Make sure torch is the CPU wheel (see note below)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD ["uvicorn", "app5:app", "--host", "0.0.0.0", "--port", "8080"]
