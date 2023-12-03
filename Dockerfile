RUN apk update
RUN apk add --no-cache --update \
	py3-pip bash g++\
    python3 python3-dev gcc \
    gfortran musl-dev \
    libffi-dev openssl-dev


ADD ./requirements.txt /tmp/requirements.txt
# Install dependencies
RUN pip3 install --no-cache-dir -q -r /tmp/requirements.txt


FROM tensorflow/serving
COPY classifier/saved_models classifier/saved_models

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku
CMD -p 8501:8501 --name tfserving_classifier --mount type=bind,source=C:/Users/ollie/OneDrive/Documents/Coding/UKPN/UKPN_ML_app/classifier/saved_models,target=/models/classifier -e MODEL_NAME=classifier -t tensorflow/serving


#tensorflow_model_server --port=8500 --rest_api_port="${PORT}" --model_name="${MODEL_NAME}" --model_base_path="${MODEL_BASE_PATH}"/"${MODEL_NAME}" "$@"