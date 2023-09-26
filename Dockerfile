# syntax=docker/dockerfile:1.2
FROM python:3.9-slim
# put you docker configuration here

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Specify the command to run on container start
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]