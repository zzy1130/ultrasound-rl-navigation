# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy ALL source code first (needed for editable install)
COPY . .

# Install PyTorch (CPU version) and project dependencies
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system -e .

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]

