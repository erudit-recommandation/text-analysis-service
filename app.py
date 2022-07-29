from flask import Flask, render_template, request, redirect, url_for

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
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

SECRET_KEY = os.getenv("SECRET")

model_path = Path("./models/d2v.model")
model = None 

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


@app.route("/")
def hello_world():
    if model is not None:
        return "Le service d'analyse de texte est en fonction"
    else:
        return "Le service d'analyse de texte est en fonction, mais le modèle n'est pas définit contacter l'administrateur pour utiliser le service"


@app.route("/gemsim", methods=['POST'])
def gemsim():
    if model is None:
        return "Le modèle n'est pas définit contacter l'administrator", 500
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


@app.route("/model", methods=['POST'])
def post_models():
    global model
    d2v_file = request.files['d2v.model']
    syn1neg_file = request.files['d2v.model.syn1neg.npy']
    vector_file = request.files['d2v.model.wv.vectors.npy']

    if request.form['password'] != SECRET_KEY:
        return "le mot de passe n'est pas valide", 400

    if d2v_file.filename != 'd2v.model' or syn1neg_file.filename != 'd2v.model.syn1neg.npy' or vector_file.filename != 'd2v.model.wv.vectors.npy':
        return "pas tous les fichiers ont été définit", 400

    d2v_file.save("models/{}".format(d2v_file.filename))
    syn1neg_file.save("models/{}".format(syn1neg_file.filename))
    vector_file.save("models/{}".format(vector_file.filename))

    model = Word2Vec.load(str(model_path))

    return "Le modèle à été changer"


@app.route("/model", methods=['GET'])
def render_models_page():
    return render_template('model.html')


if __name__ == "__main__":
    if os.path.exists(model_path):
        model = Word2Vec.load(str(model_path))
    print("server ready")
    from waitress import serve
    serve(app, host="0.0.0.0", port=8092)
