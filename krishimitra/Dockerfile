FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Train models on first run (fetches real weather data)
RUN python scripts/train_all.py || echo "Training skipped (will use KB fallback)"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health').raise_for_status()" || exit 1

CMD ["python", "api/main.py"]
