FROM python:3.12-slim
COPY requirements.txt /tmp/
RUN apt-get install gcc && pip install -r /tmp/requirements.txt
