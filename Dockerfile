FROM --platform=linux/amd64 python:3.8

WORKDIR /monkey-model

COPY . ./

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx

RUN pip install -r requirements.txt

CMD [ "python", "app.py"]