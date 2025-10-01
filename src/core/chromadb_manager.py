"""
ChromaDB Manager
Handles all ChromaDB operations for storing and retrieving chunked documents with embeddings
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
from typing import List, Dict, Any, Optional


class ChromaDBManager:
    """
    Manager class for ChromaDB operations with embedding support
    """
    
    def __init__(self, 
                 collection_name: str, 
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_custom_embeddings: bool = False):
        """
        Initialize ChromaDB manager with embedding function
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist ChromaDB data
            embedding_model (str): Name of the embedding model to use
                Options:
                - "all-MiniLM-L6-v2" (default, fast, good quality)
                - "all-mpnet-base-v2" (slower, better quality)
                - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
            use_custom_embeddings (bool): If True, you must provide embeddings manually
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.use_custom_embeddings = use_custom_embeddings
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Set up embedding function
        if not use_custom_embeddings:
            # Use sentence transformers for automatic embeddings
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            print(f"Using embedding model: {embedding_model}")
        else:
            self.embedding_function = None
            print("Custom embeddings mode enabled - you must provide embeddings")
        
        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Crawled web content for RAG with embeddings"}
        )
        
        print(f"ChromaDB initialized:")
        print(f"  - Persist directory: {persist_directory}")
        print(f"  - Collection: {collection_name}")
        print(f"  - Existing chunks: {self.collection.count()}")
    
    def add_documents(self, 
                     chunked_documents: List[Dict[str, Any]], 
                     embeddings: Optional[List[List[float]]] = None,
                     batch_size: int = 100):
        """
        Add chunked documents to ChromaDB with automatic or custom embeddings
        
        Args:
            chunked_documents (list): List of chunked documents with metadata
            embeddings (list, optional): Pre-computed embeddings (if use_custom_embeddings=True)
            batch_size (int): Number of documents to add per batch (default: 100)
        """
        if not chunked_documents:
            print("No documents to add")
            return
        
        if self.use_custom_embeddings and embeddings is None:
            raise ValueError("Custom embeddings mode enabled but no embeddings provided")
        
        if embeddings and len(embeddings) != len(chunked_documents):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of documents ({len(chunked_documents)})")
        
        total_docs = len(chunked_documents)
        print(f"\nAdding {total_docs} chunks to ChromaDB with embeddings...")
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch = chunked_documents[i:i + batch_size]
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            batch_embeddings = None
            
            for idx, doc in enumerate(batch):
                # Document text
                documents.append(doc['chunk_text'])
                
                # Metadata (ChromaDB requires all values to be strings, ints, or floats)
                metadata = {
                    'chunk_index': doc['chunk_index'],
                    'total_chunks': doc['total_chunks'],
                    'url': doc['url'],
                    'depth': doc['depth'],
                    'parent_url': str(doc['parent_url']) if doc['parent_url'] else '',
                    'title': doc['title'],
                    'description': doc['description'],
                    'content_type': doc['content_type'],
                    'original_text_length': doc['original_text_length']
                }
                metadatas.append(metadata)
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
            
            # Get embeddings for this batch if custom embeddings provided
            if embeddings:
                batch_embeddings = embeddings[i:i + batch_size]
            
            # Add batch to ChromaDB
            add_params = {
                'documents': documents,
                'metadatas': metadatas,
                'ids': ids
            }
            
            # Add embeddings if provided
            if batch_embeddings:
                add_params['embeddings'] = batch_embeddings
            
            self.collection.add(**add_params)
            
            print(f"  Added batch {i // batch_size + 1}: {len(documents)} chunks with embeddings")
        
        print(f"\nSuccessfully added {total_docs} chunks with embeddings to ChromaDB")
    
    def query(self, 
              query_text: str = None,
              query_embedding: List[float] = None,
              n_results: int = 5, 
              filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Query ChromaDB for similar documents using text or embedding
        
        Args:
            query_text (str): Query text (if not providing embedding)
            query_embedding (list): Query embedding vector (if not providing text)
            n_results (int): Number of results to return (default: 5)
            filter_metadata (dict): Optional metadata filters (e.g., {'depth': 0})
            
        Returns:
            dict: Query results with documents and metadata
        """
        if query_text is None and query_embedding is None:
            raise ValueError("Must provide either query_text or query_embedding")
        
        query_params = {
            'n_results': n_results
        }
        
        if query_text:
            query_params['query_texts'] = [query_text]
        elif query_embedding:
            query_params['query_embeddings'] = [query_embedding]
        
        if filter_metadata:
            query_params['where'] = filter_metadata
        
        results = self.collection.query(**query_params)
        return results
    
    def get_by_url(self, url: str, n_results: int = 10) -> Dict:
        """
        Get all chunks from a specific URL
        
        Args:
            url (str): URL to filter by
            n_results (int): Maximum number of results
            
        Returns:
            dict: Query results
        """
        results = self.collection.get(
            where={"url": url},
            limit=n_results
        )
        return results
    
    def delete_collection(self):
        """
        Delete the entire collection
        Warning: This will permanently delete all data in the collection
        """
        self.client.delete_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' deleted")
    
    def reset_collection(self):
        """
        Reset the collection (delete and recreate)
        """
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Crawled web content for RAG with embeddings"}
        )
        print(f"Collection '{self.collection_name}' reset")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            dict: Collection statistics
        """
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'embedding_model': self.embedding_model if not self.use_custom_embeddings else 'custom',
            'persist_directory': self.persist_directory
        }
    
    def print_query_results(self, results: Dict, max_preview_length: int = 200):
        """
        Pretty print query results
        
        Args:
            results (dict): Query results from ChromaDB
            max_preview_length (int): Maximum length of text preview
        """
        if not results['documents'] or not results['documents'][0]:
            print("No results found")
            return
        
        print(f"\nFound {len(results['documents'][0])} results:")
        print("=" * 80)
        
        # Get distances if available
        distances = results.get('distances', [[]])[0] if 'distances' in results else None
        
        for idx, (doc, metadata) in enumerate(zip(results['documents'][0], 
                                                   results['metadatas'][0]), 1):
            print(f"\nResult {idx}:")
            if distances:
                print(f"  Similarity Score: {1 - distances[idx-1]:.4f}")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  URL: {metadata.get('url', 'N/A')}")
            print(f"  Depth: {metadata.get('depth', 'N/A')}")
            print(f"  Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
            print(f"  Content Type: {metadata.get('content_type', 'N/A')}")
            print(f"  Preview: {doc[:max_preview_length]}...")
            print("-" * 80)