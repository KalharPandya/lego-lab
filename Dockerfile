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

# Copy the rest of the application files into the container
COPY . .

# Expose ports on which your Gradio apps will run
# e.g. 80 for app.py, 7890 for sticker.py
EXPOSE 80
EXPOSE 7890

# Run both servers in the same container
# The '&' runs them in the background, and 'wait -n' ensures the container
# terminates if either process exits with an error.
CMD ["/bin/bash", "-c", "python app.py& \
                         python sticker.py& \
                         wait -n"]
