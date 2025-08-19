FROM python:3.11-slim

# Install runtime libs for OpenCV and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 5000

ENV OLLAMA_API_BASE=http://ollama:11434
ENV OLLAMA_MODEL=mistral

CMD ["python", "main.py"]
