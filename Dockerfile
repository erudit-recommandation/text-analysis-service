FROM python:3

RUN mkdir /app

ADD . /app/

WORKDIR /app

RUN pip3 install -r requirement.txt

EXPOSE 8092

CMD [ "python3", "app.py"]

