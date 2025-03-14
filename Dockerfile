# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python buffering and enable UTF-8
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose ports for Gradio apps
EXPOSE 7891 7890

# Run both servers in parallel and keep the container alive
CMD ["bash", "-c", "python app.py & python sticker.py & wait"]
