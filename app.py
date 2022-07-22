from flask import Flask, request

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Word2Vec
from gnomonics.preprocessing import lemmatizing_text
from nltk.tokenize import word_tokenize
from pathlib import Path
import nltk
import json
import argparse
import os
import time


app = Flask(__name__)

"""
convert an input text to an inferance vector
"""


def convert_to_inf_vec(text):
    inf_txt = "la crise de l'environnement"

    inf_vec = lemmatizing_text(inf_txt.split(' '), language='french')
    inf_vec = [l for l in inf_vec if len(l) > 0]
    return inf_vec


"""
get the recommandation of the model it return the index and the related score

list(tuple(int, float))
"""


def get_recommandations(inf_vec, n):
    global model
    recommandation_indexes = model.dv.similar_by_vector(
        model.infer_vector(inf_vec), topn=n)  # ,restrict_vocab=10000)
    return recommandation_indexes[:n]


"""
convert the recommandation into a json
{
    "{index}": score(float)
}
"""


def convert_to_json(recommandation_indexes):
    map_recommandation = {}
    for rec in recommandation_indexes:
        map_recommandation[rec[0]] = rec[1]
    json_recommandation = json.dumps(map_recommandation)
    return json_recommandation


model_path = Path("./models/d2v_articles/d2v.model")
model = Word2Vec.load(str(model_path))


@app.route("/")
def hello_world():
    return "hello world!"


@app.route("/gemsim", methods=['POST'])
def gemsim():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        text = json["text"]
        n = json["n"]
        print("-- query: {} --".format(text))
        inf_vec = convert_to_inf_vec(text=text)
        recommandation_indexes = get_recommandations(inf_vec=inf_vec, n=n)
        result = convert_to_json(recommandation_indexes)
        print("-- result: {} --".format(result))
        return result
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8092)
