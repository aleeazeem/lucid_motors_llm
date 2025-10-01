"""
Embedding Generator using Sentence Transformers
Minimal implementation for generating embeddings with multi-threading
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers with multi-threading support"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name (str): Sentence transformer model name
            device (str): 'cuda', 'cpu', or None for auto-detect
        """
        print(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.lock = threading.Lock()
        
        print(f"✓ Model loaded - Dimension: {self.embedding_dim}, Device: {self.model.device}")
    
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    
    def _generate_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Thread-safe batch embedding generation"""
        with self.lock:
            embeddings = self.model.encode(texts, batch_size=batch_size, 
                                          show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()
    
    
    def generate_embeddings_for_chunks(self, 
                                      chunked_documents: List[Dict[str, Any]],
                                      batch_size: int = 32,
                                      use_multithreading: bool = True,
                                      num_workers: int = 4,
                                      show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for chunked documents
        
        Args:
            chunked_documents: List of dicts with 'chunk_text' field
            batch_size: Batch size for processing
            use_multithreading: Enable multi-threading
            num_workers: Number of worker threads
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        texts = [doc['chunk_text'] for doc in chunked_documents]
        
        if not use_multithreading or len(texts) <= batch_size:
            # Single-threaded
            embeddings = self.model.encode(texts, batch_size=batch_size,
                                          show_progress_bar=show_progress,
                                          convert_to_numpy=True)
            return embeddings.tolist()
        
        # Multi-threaded
        chunk_size = max(batch_size, len(texts) // num_workers)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        all_embeddings = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._generate_batch, chunk, batch_size): (idx, len(chunk))
                for idx, chunk in enumerate(text_chunks)
            }
            
            with tqdm(total=len(texts), desc="Generating embeddings", disable=not show_progress) as pbar:
                for future in as_completed(future_to_idx):
                    chunk_idx, chunk_len = future_to_idx[future]
                    embeddings = future.result()
                    start_idx = chunk_idx * chunk_size
                    for i, emb in enumerate(embeddings):
                        all_embeddings[start_idx + i] = emb
                    pbar.update(chunk_len)
        
        return all_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': str(self.model.device),
            'max_seq_length': self.model.max_seq_length
        }
    