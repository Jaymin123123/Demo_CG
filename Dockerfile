FROM python:3.11-slim

# System deps for PDFs/OCR
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Faster layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + data
COPY . .

# Fly will map traffic to this port; Uvicorn will listen here
ENV PORT=8080

# If your FastAPI object is called `app` inside app5.py:
CMD ["uvicorn", "app5:app", "--host", "0.0.0.0", "--port", "8080"]
