FROM tensorflow/serving

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME classifier

COPY models/classifier /models/classifier

# Fix because base tf_serving_entrypoint.sh does not take $PORT env variable while $PORT is set by Heroku
# CMD is required to run on Heroku
COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh
RUN chmod +x /usr/bin/tf_serving_entrypoint.sh
ENTRYPOINT []
CMD ["/usr/bin/tf_serving_entrypoint.sh"]


tensorflow_model_server --port=8500 --rest_api_port="${PORT}" --model_name="${MODEL_NAME}" --model_base_path="${MODEL_BASE_PATH}"/"${MODEL_NAME}" "$@"
