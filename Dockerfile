#Grab the latest alpine image
FROM python:3.9.18-slim



ADD ./requirements.txt /tmp/requirements.txt

#make virt env because otherwise you get some overlap or something
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -q -r /tmp/requirements.txt
RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/lib/nltk_data')" ]


# Add our code
ADD ./webapp /opt/webapp/
COPY ./classifier /opt/webapp/classifier

WORKDIR /opt/webapp


# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku			
CMD gunicorn --bind 0.0.0.0:$PORT wsgi
