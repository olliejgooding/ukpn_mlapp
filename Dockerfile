# Use an official TensorFlow Serving runtime as a parent image
FROM tensorflow/serving:latest

# First stage: Build dependencies and install Python packages
#FROM python:3.11.2-slim AS builder

COPY ./requirements.txt /requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt

# Create the model directory and copy the saved model
RUN mkdir -p /models/classifier
COPY classifier/saved_models /models/classifier


# Expose the port tensorflow serving will run on
EXPOSE 8501

# Define environment variable
ENV MODEL_NAME=classifier

# Command to run on container start
CMD ["tensorflow_model_server", "--port=8501", "--rest_api_port=8502", "--model_name=${MODEL_NAME}", "--model_base_path=/models/${MODEL_NAME}"]
