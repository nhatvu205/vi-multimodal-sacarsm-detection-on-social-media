# Dockerfile for the helper scripts service.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies as a separate layer so Docker cache reuses it
# unless scripts/requirements.txt actually changes.
COPY scripts/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
