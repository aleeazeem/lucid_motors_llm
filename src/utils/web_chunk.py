"""
Text Chunking Utilities
Provides text chunking functionality with overlap for RAG applications
"""

from typing import List, Dict, Any


class WebTextChunker:
    """
    Utility class for chunking text into overlapping segments
    """
    
    def __init__(self, chunk_size_words, overlap_percentage):
        """
        Initialize the text chunker
        
        Args:
            chunk_size_words (int): Number of words per chunk (default: 500)
            overlap_percentage (int): Percentage of overlap between chunks (default: 15)
        """
        self.chunk_size_words = chunk_size_words
        self.overlap_percentage = overlap_percentage
        self.overlap_words = int(chunk_size_words * overlap_percentage / 100)
        
        print(f"TextChunker initialized:")
        print(f"  - Chunk size: {chunk_size_words} words")
        print(f"  - Overlap: {overlap_percentage}% ({self.overlap_words} words)")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text (str): Text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split text into words
        words = text.split()
        
        # If text is shorter than chunk size, return as single chunk
        if len(words) <= self.chunk_size_words:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            # Get chunk of words
            end = start + self.chunk_size_words
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Move start position (with overlap)
            start = end - self.overlap_words
            
            # Break if we've reached the end
            if end >= len(words):
                break
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document and preserve metadata
        
        Args:
            document (dict): Document with 'text' and metadata fields
            
        Returns:
            List[dict]: List of chunked documents with metadata
        """
        text = document.get('text', '')
        chunks = self.chunk_text(text)
        
        chunked_documents = []
        for idx, chunk in enumerate(chunks):
            chunked_doc = {
                'chunk_text': chunk,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'url': document.get('url', ''),
                'depth': document.get('depth', 0),
                'parent_url': document.get('parent_url', ''),
                'title': document.get('title', ''),
                'description': document.get('description', ''),
                'content_type': document.get('content_type', 'html'),
                'original_text_length': document.get('text_length', 0)
            }
            chunked_documents.append(chunked_doc)
        
        return chunked_documents
    
    def chunk_documents_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents at once
        
        Args:
            documents (list): List of documents to chunk
            
        Returns:
            List[dict]: List of all chunked documents with metadata
        """
        all_chunks = []
        
        for doc in documents:
            # Skip documents without text (e.g., PDF references)
            if 'text' not in doc or not doc.get('text'):
                print(f"  Skipping document without text: {doc.get('title', 'Unknown')}")
                continue
            
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  Chunked '{doc.get('title', 'Untitled')}': {len(chunks)} chunks")
        # each checunk will be in the form dictionary with metadata
        return all_chunks