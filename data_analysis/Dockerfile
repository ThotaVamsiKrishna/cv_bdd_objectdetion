# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="vamsikrishna"
LABEL description="BDD100K data analysis pipeline and interactive dashboard"

# System dependencies for Pillow and Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /deps /usr/local

WORKDIR /app
COPY . .

# Pre-compute output directory
RUN mkdir -p /data/precomputed

# Dashboard port
EXPOSE 8050

# Default: run the data analysis pipeline then launch the dashboard.
# Override CMD at runtime to run only analysis or only the dashboard.
CMD ["python", "main.py", "--help"]
