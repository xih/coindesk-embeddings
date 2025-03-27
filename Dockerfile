FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p scrapers

# Copy only necessary files
COPY scrapers/rag.py scrapers/
COPY scrapers/generate_embeddings.py scrapers/
COPY trpc_backend.py .

# Environment variables
ENV OPENAI_API_KEY=""
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Command to run FastAPI server
CMD uvicorn trpc_backend:app --host 0.0.0.0 --port ${PORT:-8000} 