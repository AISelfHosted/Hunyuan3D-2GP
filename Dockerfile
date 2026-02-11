FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the hy3dgen package
RUN pip3 install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/logs /app/gradio_cache

# Expose ports
# 8080 = Gradio UI, 8081 = API Server
EXPOSE 8080 8081

# Default to Gradio app
ENV APP_MODE=gradio
ENV EXTRA_ARGS=""

# Use a shell entrypoint to support mode switching
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["if [ \"$APP_MODE\" = 'api' ]; then \
       python3 api_server.py --host 0.0.0.0 --port 8081 $EXTRA_ARGS; \
     else \
       python3 gradio_app.py --host 0.0.0.0 --port 8080 $EXTRA_ARGS; \
     fi"]
