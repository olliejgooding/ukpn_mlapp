# First stage: Build dependencies and install Python packages
FROM tensorflow/tensorflow:latest AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev gcc gfortran libffi-dev openssl-dev

COPY ./requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Second stage: Copy only necessary files and the installed Python packages
FROM tensorflow/serving

COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

COPY classifier/saved_models /models/classifier

CMD ["tensorflow_model_server", "--port=8501", "--rest_api_port=8502", "--model_name=classifier", "--model_base_path=/models/classifier"]
