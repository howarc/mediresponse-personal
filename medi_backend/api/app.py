from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import awsgi
import os
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convo import generate_intro, generate_convo
app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate_response():    
    data = request.get_json()
    if (data.get("isIntro") == True):
        context, rel_response, doc_response  = generate_intro()
        return jsonify({"context": context, "rel_response": rel_response, "doc_response": doc_response})
    rel_response, doc_response = generate_convo(data.get("inputText"))
    return jsonify({"rel_response": rel_response, "doc_response": doc_response})

def lambda_handler(event, context):
    return awsgi.response(app, event, context)