FROM python:3

RUN mkdir /app

ADD . /app/

WORKDIR /app

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip3 install -r requirement.txt
RUN python3 -m spacy download fr_core_news_sm
RUN python3 init.py

EXPOSE 8092

CMD [ "python3", "app.py"]
