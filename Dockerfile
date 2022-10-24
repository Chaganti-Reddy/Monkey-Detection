FROM --platform=linux/amd64 python:3.8

WORKDIR /app

COPY ./models .models/

COPY ./results ./results

COPY ./static ./static

COPY ./templates ./templates

COPY ./utils ./utils

COPY ./app.py ./app.py

COPY ./requirements.txt ./requirements.txt

COPY ./best.pt ./best.pt

COPY ./detect.py ./detect.py

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD ["app.py"]