import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json
from flask import Flask, request, jsonify
from config import Config
from SimiliarService import SimiliarService
from flask_cors import CORS
import logging

app = Flask(__name__)

CORS(app , resources = {r"/*" : {"origins" :["https://localhost:7282", "*"]}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading Similarity Model ....")
ModelSimilarity = SimiliarService()
logger.info("Model Loading Successfully")

@app.route("/health",methods= ["Get"])
def HealthCheck():
    return jsonify({"status": "OK", "service" : "Similarity API"}), 200


@app.route("/similar", methods = ["POST"])
def FindSimilar():
    try:
        data = request.get_json()
        if not data or 'Name' not in data:
            return jsonify({"error": "Missing 'Name' in request body."}), 400
        
        name = data['Name']
        threshold = data.get('threshold', Config.DefaultThreshold)
    
        results = ModelSimilarity.find_similar(name, threshold)
        
        if isinstance(results, tuple):
            return jsonify(results[0]), results[1]
        
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 


if __name__ == '__main__':
    
    app.run(debug=True,port = 9000)
    
    
            
        
    



