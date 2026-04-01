FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (including git for cloning/dev and python3-dev for headers)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install build dependencies (torch for C++ extension compilation)
# Note: requirements.txt installs torch, so this is handled below.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the hy3dgen package (triggers custom_rasterizer build)
# Using standard install (not editable) for cleaner container image
RUN pip3 install --no-cache-dir .

# Create directories and set permissions if needed
RUN mkdir -p /app/logs /app/launcher_cache

# Expose ports
EXPOSE 8080 8081

# Default to launcher app
ENV APP_MODE=launcher
ENV EXTRA_ARGS=""

# Entrypoint script logic inline
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["if [ \"$APP_MODE\" = 'api' ]; then \
       hy3dgen-api --host 0.0.0.0 --port 8081 $EXTRA_ARGS; \
     else \
       hy3dgen-launcher --host 0.0.0.0 --port 8080 $EXTRA_ARGS; \
     fi"]
