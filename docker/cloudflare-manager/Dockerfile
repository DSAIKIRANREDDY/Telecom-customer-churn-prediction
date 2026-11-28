
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Download the Cloudflare Tunnel binary.
ARG TARGETARCH

RUN case "${TARGETARCH}" in \
        amd64) ARCH="amd64" ;; \
        arm64) ARCH="arm64" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    curl -fsSL \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}" \
        -o /usr/local/bin/cloudflared && \
    chmod +x /usr/local/bin/cloudflared

# Install Python dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the tunnel manager.
COPY tunnel_manager.py .

# Start the manager.
CMD ["python", "-u", "tunnel_manager.py"]
