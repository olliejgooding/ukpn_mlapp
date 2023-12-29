FROM tensorflow/serving:latest-devel


ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=0

# Create the model directory and copy the saved model
RUN mkdir -p /models/classifier
COPY classifier/saved_models /models/classifier


RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=classifier --model_base_path=/models/classifier \
--tensorflow_session_parallelism=1 \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh


# Run the image as a non-root user
RUN adduser myuser1
USER myuser1


ENTRYPOINT []
CMD ["/usr/bin/tf_serving_entrypoint.sh"]