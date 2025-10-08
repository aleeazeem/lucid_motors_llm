# text_chunker.py
from typing import List, Dict, Any

class TextChunker:
    """Fixed-size text chunker with overlap"""
    
    def __init__(self, chunk_size_words: int = 500, overlap_percentage: int = 15):
        self.chunk_size_words = chunk_size_words
        self.overlap_percentage = overlap_percentage
        self.overlap_words = int(chunk_size_words * overlap_percentage / 100)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= self.chunk_size_words:
            return [text]
        
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size_words
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - self.overlap_words
            if end >= len(words):
                break
        return chunks
    
    def chunk_documents_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents"""
        all_chunks = []
        for doc in documents:
            if 'text' not in doc or not doc['text']:
                continue
            
            text_chunks = self.chunk_text(doc['text'])
            for idx, chunk in enumerate(text_chunks):
                all_chunks.append({
                    'chunk_text': chunk,
                    'chunk_index': idx,
                    'total_chunks': len(text_chunks),
                    'url': doc.get('url', ''),
                    'depth': doc.get('depth', 0),
                    'parent_url': doc.get('parent_url', ''),
                    'title': doc.get('title', ''),
                    'description': doc.get('description', ''),
                    'content_type': doc.get('content_type', 'html'),
                    'original_text_length': len(doc['text'])
                })
        return all_chunks