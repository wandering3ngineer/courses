# Use the official Python image from the Docker Hub
FROM python:3.11.9-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake gcc g++ clang libomp-dev
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code
COPY . /app

# Expose the necessary ports. In this case when we 
# make a request of the llm we do so 
EXPOSE 8002

# Command to run the application in the shell
# sh -c <string> where <string> = "python llm.py"
CMD ["sh", "-c", "python llm.py"]
