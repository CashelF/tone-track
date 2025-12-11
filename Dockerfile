# Container image for hosting the Flask backend on Hugging Face Spaces
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install Python dependencies (CPU wheels for PyTorch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code
COPY . .

EXPOSE ${PORT}

# Run the Flask app (Hugging Face Spaces expects the service on 0.0.0.0:${PORT})
CMD ["python", "app.py"]
