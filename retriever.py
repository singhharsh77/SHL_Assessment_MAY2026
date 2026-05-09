import json
import logging
import os
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TEST_TYPE_MAP = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Competencies": "C",
    "Biodata & Situational Judgment": "S",
    "Assessment Exercises": "S",
    "Development & 360": "C"
}

class HybridRetriever:
    def __init__(self, json_path="shl_data.json"):
        self.items = []
        self.texts = []
        
        # Load and format the data
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f, strict=False)
                
            for idx, d in enumerate(raw_data):
                keys = d.get("keys", [])
                primary_key = keys[0] if keys else ""
                test_type = TEST_TYPE_MAP.get(primary_key, "K")
                
                # Keep a clean dictionary for the final response
                clean_item = {
                    "id": idx,
                    "name": d.get("name", ""),
                    "url": d.get("link", ""),
                    "test_type": test_type,
                    "desc": d.get("description", ""),
                    "levels": d.get("job_levels", []),
                    "keys": keys
                }
                self.items.append(clean_item)
                
                # Create Structured Natural Language Representation
                structured_text = f"[ID: {idx}]\n"
                structured_text += f"Name: {clean_item['name']}\n"
                structured_text += f"URL: {clean_item['url']}\n"
                structured_text += f"TestType_Code: {test_type}\n"
                structured_text += f"Type: {', '.join(keys)}\n"
                structured_text += f"Job Levels: {', '.join(clean_item['levels'])}\n"
                structured_text += f"Description: {clean_item['desc']}\n"
                self.texts.append(structured_text)
                
            logger.info(f"Formatted {len(self.items)} catalog items.")
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            
        # Initialize MiniLM and compute embeddings
        try:
            logger.info("Initializing MiniLM embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Simple caching to avoid recomputing on every hot reload during dev
            cache_file = 'shl_embeddings.npy'
            if os.path.exists(cache_file):
                logger.info("Loading cached embeddings...")
                self.embeddings = np.load(cache_file)
            else:
                logger.info("Computing embeddings (this takes ~2 seconds)...")
                self.embeddings = self.model.encode(self.texts, show_progress_bar=False)
                np.save(cache_file, self.embeddings)
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.model = None

    def search(self, query: str, top_k: int = 50):
        if not self.items or self.model is None:
            return []
            
        # Embed query and compute cosine similarity
        query_embedding = self.model.encode([query])[0]
        
        # Cosine similarity (embeddings are already normalized by sentence-transformers typically, but we dot product)
        # to be safe, compute actual cosine similarity
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        norm_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(norm_embeddings, norm_query)
        
        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(self.texts[idx])
            
        return results
        
    def get_item_by_name(self, name: str):
        for item in self.items:
            if item["name"] == name:
                return item
        return None

# Singleton instance
retriever = HybridRetriever()
