# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables to prevent Python buffering and enable UTF-8
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container,
# including the final_model_quantized.pth file.
COPY . .

# Expose the port on which Gradio will run (default is 7860)
EXPOSE 80

# Command to run the Gradio app
CMD ["python", "app.py"]
