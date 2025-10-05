# Multi-stage build for production deployment
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .

# Install the package
RUN pip install -e .

# API service stage
FROM base as api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Web interface stage  
FROM base as web
EXPOSE 8501
CMD ["streamlit", "run", "src/web/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Development stage
FROM base as dev
RUN pip install pytest pytest-cov black flake8 jupyter
CMD ["bash"]