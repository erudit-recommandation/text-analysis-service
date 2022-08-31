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
from dotenv import dotenv_values
import operator

import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', required=False,
                    action="store_true", help="developpement mode")

parser.add_argument('-p', required=False,
                    action="store_true", help="production mode")
args, _ = parser.parse_known_args()
config = None
if args.d:
    print("developpement mode")
    config = dotenv_values(".env_dev")
elif args.p:
    print("production mode")
    config = dotenv_values(".env")
else:
    print("developpement mode")
    config = dotenv_values(".env_dev")

SECRET_KEY = config["SECRET"]

models = {}
FR = "french"
ENG = "english"
for file in os.listdir("./models"):
    d = os.path.join("./models", file)
    if os.path.isdir(d):
        model_path_fr = Path(os.path.join(d, "d2v_{}.model".format(FR)))
        model_path_eng = Path(os.path.join(d, "d2v_{}.model".format(ENG)))
        if os.path.exists(model_path_fr):
            print("loading {}".format(model_path_fr))
            models[model_path_fr.parts[-2]
                   ] = (Word2Vec.load(str(model_path_fr)), FR)
        elif os.path.exists(model_path_eng):
            print("loading {}".format(model_path_eng))
            models[model_path_eng.parts[-2]
                   ] = (Word2Vec.load(str(model_path_eng)), ENG)


"""
convert an input text to an inferance vector
"""


def convert_to_inf_vec(text, language):
    print("--{}--".format(language))
    inf_vec = lemmatizing_text(text.split(' '), language=language)
    inf_vec = [l for l in inf_vec if len(l) > 0]
    return inf_vec


"""
get the recommandation of the model it return the index and the related score

list(tuple(int, float))
"""


def get_recommandations(inf_vec, n, corpus):
    global models
    model =  models[corpus][0]
    recommandation_indexes =model.dv.similar_by_vector(
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
    return map_recommandation


@app.route("/")
def hello_world():
    if len(models) != 0:
        return "Le service d'analyse de texte est en fonction"
    else:
        return "Le service d'analyse de texte est en fonction, mais le modèle n'est pas définit contacter l'administrateur pour utiliser le service"


@app.route("/gensim", methods=['POST'], strict_slashes=False)
def gensim():
    if models is None:
        return "Le modèle n'est pas définit contacter l'administrator", 500
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        requestJson = request.json
        text = requestJson["text"]
        n = requestJson["n"]
        corpus = requestJson["corpus"]
        print("-- query: {} --".format(text))
        inf_vec = convert_to_inf_vec(text=text, language=models[corpus][1])
        recommandation_indexes = get_recommandations(
            inf_vec=inf_vec, n=n, corpus=corpus)
        result = convert_to_json(recommandation_indexes)
        print("-- result: {} --".format(result))
        response = app.response_class(
            response=json.dumps(result),
            mimetype='application/json'
        )
        return response
    else:
        return 'Content-Type not supported!'


@app.route("/model/<string:corpus>", methods=['POST'], strict_slashes=False)
def post_models(corpus):
    global models
    d2v_file = None
    syn1neg_file = None
    vector_file = None
    language = None
    if request.form.get("password") is not None:
        if request.form['password'] != SECRET_KEY:
            return "le mot de passe n'est pas valide", 401
    else:
        return "le mot de passe doit être présent", 401
    if request.form.get("language") is not None:
        if request.form["language"] != ENG and request.form["language"] != FR:
            return "On ne supporte que le français (french) et l'anglais (english) pour le moment; reçus {}".format(request.form["language"]), 401
        else:
            language = request.form["language"]
    else:
        return "la langue doit être spécifiée", 401

    if request.files.get('d2v.model') is not None:
        if request.files.get('d2v.model').filename != "d2v.model":
            return "le modèle n'a pas été envoyé", 401
        else:
            d2v_file = request.files['d2v.model']

    else:
        d2v_file = request.files['d2v.model']
    if operator.xor(request.files.get('d2v.model.syn1neg.npy') is None, request.files.get('d2v.model.wv.vectors.npy') is None):
        return "les fichiers npy doivent être présent, si un est présent", 401
    elif request.files.get('d2v.model.syn1neg.npy') is not None and request.files.get('d2v.model.wv.vectors.npy') is not None:
        if request.files.get('d2v.model.wv.vectors.npy').filename != 'd2v.model.wv.vectors.npy' or request.files.get('d2v.model.syn1neg.npy').filename != 'd2v.model.syn1neg.npy':
            return "les fichiers npy doivent avoir les bon noms", 401
        else:
            syn1neg_file = request.files['d2v.model.syn1neg.npy']
            vector_file = request.files['d2v.model.wv.vectors.npy']

    if not os.path.exists(Path("./models/{}".format(corpus))):
        os.makedirs(Path("./models/{}".format(corpus)))

    d2v_file.save("models/{}/d2v_{}.model".format(corpus, language))
    if syn1neg_file is not None:
        syn1neg_file.save(
            "models/{}/d2v_{}.model.syn1neg.npy".format(corpus, language))
        vector_file.save(
            "models/{}/d2v_{}.model.wv.vectors.npy".format(corpus, language))

    models[corpus] = Word2Vec.load(
        str("models/{}/d2v_{}.model".format(corpus, language)))

    return "Le modèle {} à été changer".format(corpus)


@app.route("/model", methods=['GET'], strict_slashes=False)
def map_models():
    corpus = list(models.keys())
    response = app.response_class(
        response=json.dumps({"payload": corpus}),
        mimetype='application/json'
    )
    return response


@app.route("/model/<string:corpus>", methods=['GET'], strict_slashes=False)
def render_models_page(corpus):
    args = request.args
    largerModel = args.get("larger")
    if largerModel == "false":
        largerModel = False
    else:
        largerModel = True

    return render_template('model.html', corpus=corpus, largerModel=largerModel)


if __name__ == "__main__":
    print("server ready")
    from waitress import serve
    serve(app, host="0.0.0.0", port=8092)
