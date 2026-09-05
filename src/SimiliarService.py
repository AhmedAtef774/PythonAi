import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from config import Config
from Loader import LoadData

class SimiliarService:
    def __init__(self):
        self.model = SentenceTransformer(Config.Model)
        self.df = LoadData()

    def find_similar(self, Name, threshold=Config.DefaultThreshold):
        if self.df.empty:
            return {"error": "No data found in the JSON file."}, 404
        
        words = re.split(r'\s+', Name.lower())
        mask = self.df['Name'].str.contains('|'.join(words), case=False, na=False)
        candidates = self.df[mask]
        
        if candidates.empty:
            return {"error": "No products found matching the input name."}, 404
        
        query_embedding = self.model.encode([Name])
        candidate_embeddings = self.model.encode(candidates['Name'].tolist())
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        candidates = candidates.copy()
        candidates['similarity'] = similarities
        results = candidates[candidates['similarity'] >= threshold].sort_values(by='similarity', ascending=False)
        
        if results.empty:
            return {"error": "No products found above the similarity threshold."}, 404
        
        results = results.to_dict(orient='records')
        
        return results

