# pin to stable so apt packages exist
FROM python:3.11-slim-bookworm

# If you don't actually need OCR, you can drop tesseract-ocr entirely.
# PyMuPDF and pdfminer.six generally don't need extra system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD ["uvicorn", "app5:app", "--host", "0.0.0.0", "--port", "8080"]
