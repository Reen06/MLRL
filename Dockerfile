FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies.
RUN pip install --no-cache-dir scikit-learn

# Copy the entire project into the image.
COPY . /app

# Ensure the config directory exists with appropriate permissions.
RUN mkdir -p config

CMD ["python", "src/cli.py"]

