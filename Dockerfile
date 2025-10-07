# QuantumBCI - Production-Ready Dockerfile
# Streamlit + PostgreSQL + Quantum ML Application

FROM python:3.11-slim-bookworm

# Environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    postgresql-client \
    libpq-dev \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt* ./

# Install Python packages
RUN pip install --no-cache-dir \
    streamlit==1.40.0 \
    bcrypt==4.2.1 \
    cryptography==44.0.0 \
    fastapi==0.115.6 \
    fpdf==1.7.2 \
    matplotlib==3.9.3 \
    mne==1.8.0 \
    pennylane==0.39.0 \
    plotly==5.24.1 \
    pyedflib==0.1.38 \
    scikit-learn==1.6.0 \
    scipy==1.14.1 \
    streamlit-autorefresh==1.0.1 \
    tensorflow-quantum==0.7.3 \
    uvicorn==0.32.1 \
    websockets==14.1 \
    psycopg2-binary==2.9.10 \
    numpy==1.26.4 \
    pandas==2.2.3

# Create Streamlit config directory
RUN mkdir -p /app/.streamlit

# Copy Streamlit config
COPY .streamlit/config.toml /app/.streamlit/

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

# Switch to non-root user
USER streamlit

# Expose port 5000 (as per QuantumBCI configuration)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:5000/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "app.py", \
    "--server.port=5000", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=true"]
