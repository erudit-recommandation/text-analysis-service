FROM python:3.9.2

WORKDIR python-docker

COPY requirement.txt requirement.txt

RUN pip3 install -r requirement.txt

COPY . .

EXPOSE 5000


CMD [ "python3", "app.py"]