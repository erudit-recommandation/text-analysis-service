FROM python:3.9.2

WORKDIR /app

COPY requirement.txt requirement.txt

RUN pip3 install -r requirement.txt

COPY . .

EXPOSE 8092


CMD [ "python3", "app.py"]
