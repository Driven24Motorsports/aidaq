# =============================================================================
# Dockerfile — aiDAQ Cloud Deployment (Railway)
# =============================================================================
# Railway builds this image automatically when you push to GitHub.
# The container runs the Dash app on the PORT environment variable
# that Railway injects (default 8080 on Railway).
#
# Build locally to test:
#   docker build -t aidaq .
#   docker run -p 8050:8050 -e AIDAQ_MODE=cloud aidaq
# =============================================================================

# Use a slim Python 3.11 image to keep the container small
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first (Docker caches this layer separately,
# so dependencies are only re-installed when requirements.txt changes)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# NOTE: .dockerignore (if present) should exclude *.ibt, *.parquet, __pycache__
COPY config.py      .
COPY physics.py     .
COPY data_source.py .
COPY aidaq_app.py   .

# Tell the container to run in cloud mode
ENV AIDAQ_MODE=cloud

# Railway injects the PORT environment variable; Dash listens on it.
# We don't EXPOSE a fixed port — Railway handles port mapping.

# Start the app
CMD ["python", "aidaq_app.py"]
