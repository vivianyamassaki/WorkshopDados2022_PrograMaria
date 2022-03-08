FROM python:3.9.9-slim-buster

RUN mkdir /home/app

WORKDIR /home/app

RUN apt-get update \
 && pip install --no-cache-dir pylint

# install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "predictor/api.py"]