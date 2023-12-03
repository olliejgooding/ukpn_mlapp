# First stage: Build dependencies and install Python packages
FROM python:3.11.2-slim AS builder

COPY ./requirements.txt /tmp/requirements.txt

# Upgrade pip to the latest version
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Second stage: Copy only necessary files and the installed Python packages
FROM tensorflow/serving

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY classifier/saved_models /models/classifier

# Set the working directory for TensorFlow Serving
WORKDIR /models/classifier

# Expose the ports
EXPOSE 8501

# Define the command to run when the container starts
CMD ["tensorflow_model_server", "--port=8501", "--rest_api_port=8502", "--model_name=classifier", "--model_base_path=/models/classifier"]
