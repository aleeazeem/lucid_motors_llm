"""
Ingestion Pipeline
Orchestrates chunking, embedding generation, and ChromaDB storage
"""

import json
from typing import List, Dict, Any, Optional

from IPython import embed
from src.utils.web_chunk import WebTextChunker
from src.utils.embedding_generator import EmbeddingGenerator
from src.core.chromadb_manager import ChromaDBManager


class IngestionPipeline:
    """
    Complete pipeline for ingesting crawled data into ChromaDB
    Handles: JSON loading -> Chunking -> Embedding Generation -> ChromaDB Storage
    """
    
    def __init__(self,
                 chunk_size_words: int = 500,
                 overlap_percentage: int = 15,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "my_collection",
                 persist_directory: str = "./chroma_db",
                 embedding_batch_size: int = 32):
        """
        Initialize the ingestion pipeline
        
        Args:
            chunk_size_words (int): Number of words per chunk
            overlap_percentage (int): Percentage overlap between chunks
            embedding_model (str): Sentence Transformer model name
            collection_name (str): ChromaDB collection name
            persist_directory (str): Directory to store ChromaDB
            embedding_batch_size (int): Batch size for embedding generation
        """
        self.chunk_size_words = chunk_size_words
        self.overlap_percentage = overlap_percentage
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_batch_size = embedding_batch_size
        
        # Initialize components
        print("="*60)
        print("Initializing Ingestion Pipeline")
        print("="*60)
        
        # 1. Text Chunker
        print("\n[1/3] Initializing Text Chunker...")
        self.chunker = WebTextChunker(
            chunk_size_words=chunk_size_words,
            overlap_percentage=overlap_percentage
        )
        
        # 2. Embedding Generator
        print("\n[2/3] Initializing Embedding Generator...")
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model
        )
        
        # 3. ChromaDB Manager
        print("\n[3/3] Initializing ChromaDB Manager...")
        self.db_manager = ChromaDBManager(
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_custom_embeddings=True  # We're providing our own embeddings
        )
        
        print("\n" + "="*60)
        print("Ingestion Pipeline Ready!")
        print("="*60)
    
    def load_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from JSON file
        
        Args:
            json_file_path (str): Path to JSON file
            
        Returns:
            List[Dict]: List of documents
        """
        print(f"\nLoading documents from: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"✓ Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces
        
        Args:
            documents (list): List of documents with 'text' field
            
        Returns:
            List[Dict]: List of chunked documents
        """
        print("\n" + "="*60)
        print("STEP 1: Chunking Documents")
        print("="*60)
        
        chunks = self.chunker.chunk_documents_batch(documents)
        
        print(f"\n✓ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], 
                           use_multithreading: bool = True,
                           num_workers: int = 4) -> List[List[float]]:
        """
        Generate embeddings for chunks with optional multi-threading
        
        Args:
            chunks (list): List of chunked documents
            use_multithreading (bool): Enable multi-threading (default: True)
            num_workers (int): Number of worker threads (default: 4)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        print("\n" + "="*60)
        print("STEP 2: Generating Embeddings")
        print("="*60)
        
        if use_multithreading:
            print(f"Using multi-threading with {num_workers} workers")
        
        embeddings = self.embedding_generator.generate_embeddings_for_chunks(
            chunks,
            batch_size=self.embedding_batch_size,
            show_progress=True,
            use_multithreading=use_multithreading,
            num_workers=num_workers
        )
        
        print(f"\n✓ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def store_in_chromadb(self, chunks: List[Dict[str, Any]], 
                         embeddings: List[List[float]]) -> None:
        """
        Store chunks and embeddings in ChromaDB
        
        Args:
            chunks (list): List of chunked documents
            embeddings (list): List of embedding vectors
        """
        print("\n" + "="*60)
        print("STEP 3: Storing in ChromaDB")
        print("="*60)
        
        self.db_manager.add_documents(chunks, embeddings=embeddings)
        
        print(f"\n✓ Stored {len(chunks)} chunks in ChromaDB")
    
    def ingest(self, json_file_path: str, use_multithreading: bool = True,
              num_workers: int = 4) -> Dict[str, Any]:
        """
        Complete ingestion pipeline: Load -> Chunk -> Embed -> Store
        
        Args:
            json_file_path (str): Path to JSON file with crawled data
            use_multithreading (bool): Enable multi-threading for embeddings (default: True)
            num_workers (int): Number of worker threads for embeddings (default: 4)
            
        Returns:
            Dict: Statistics about the ingestion process
        """
        print("\n" + "="*80)
        print(" "*20 + "STARTING INGESTION PIPELINE")
        print("="*80)
        
        # Step 1: Load documents
        documents = self.load_json(json_file_path)
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Generate embeddings (with optional multi-threading)
        embeddings = self.generate_embeddings(chunks, use_multithreading, num_workers)
        
        # Step 4: Store in ChromaDB
        self.store_in_chromadb(chunks, embeddings)
        
        # Get statistics
        stats = self.get_statistics()
        stats['original_documents'] = len(documents)
        
        # Print summary
        print("\n" + "="*80)
        print(" "*20 + "INGESTION COMPLETE!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"  • Original documents: {stats['original_documents']}")
        print(f"  • Total chunks created: {stats['total_chunks']}")
        print(f"  • Embedding dimension: {stats['embedding_dimension']}")
        print(f"  • Multi-threading: {'Enabled' if use_multithreading else 'Disabled'}")
        if use_multithreading:
            print(f"  • Worker threads: {num_workers}")
        print(f"  • ChromaDB collection: {stats['collection_name']}")
        print(f"  • Storage location: {stats['persist_directory']}")
        print("\n" + "="*80)
        
        return stats
    
    def query(self, query_text: str, n_results: int = 5, 
              filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Query the ChromaDB collection
        
        Args:
            query_text (str): Query text
            n_results (int): Number of results to return
            filter_metadata (dict): Optional metadata filters
            
        Returns:
            Dict: Query results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        # Query ChromaDB
        results = self.db_manager.query(
            query_embedding=query_embedding,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Dict: Statistics
        """
        db_stats = self.db_manager.get_collection_stats()
        model_info = self.embedding_generator.get_model_info()
        
        return {
            'original_documents': 'N/A',  # Would need to track this separately
            'total_chunks': db_stats['total_chunks'],
            'embedding_dimension': model_info['embedding_dimension'],
            'embedding_model': model_info['model_name'],
            'collection_name': db_stats['collection_name'],
            'persist_directory': db_stats['persist_directory']
        }
    
    def reset_database(self):
        """Reset the ChromaDB collection (delete all data)"""
        print("\n⚠️  Resetting ChromaDB collection...")
        self.db_manager.reset_collection()
        print("✓ Collection reset complete")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'json_file_path': 'output/www_wellsfargo_com/www_wellsfargo_com_data.json',
        'chunk_size_words': 500,
        'overlap_percentage': 15,
        'embedding_model': 'all-MiniLM-L6-v2',
        'collection_name': 'wellsfargo_help',
        'persist_directory': './chroma_db',
        'embedding_batch_size': 32
    }
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        chunk_size_words=config['chunk_size_words'],
        overlap_percentage=config['overlap_percentage'],
        embedding_model=config['embedding_model'],
        collection_name=config['collection_name'],
        persist_directory=config['persist_directory'],
        embedding_batch_size=config['embedding_batch_size']
    )
    
    # Run ingestion
    stats = pipeline.ingest(
        config['json_file_path'],
        use_multithreading=True,  # Enable multi-threading
        num_workers=4             # Number of worker threads
    )
    
    import chromadb
    import pandas as pd
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="wellsfargo_help")

    # Get first 5 rows
    results = collection.get(limit=5, include=['documents', 'metadatas'])

    # Convert to DataFrame
    data = []
    for doc, meta in zip(results['documents'], results['metadatas']):
        data.append({
            'text_preview': doc[:15] + '...',
            'title': meta['title'][:10] + '...',
            'chunk': f"{meta['chunk_index']+1}/{meta['total_chunks']}",
            'content_type': meta['content_type']
        })
  
    df = pd.DataFrame(data)

    # Display
    print(f"\nTotal documents: {collection.count()}")
    print("\nFirst 5 rows:\n")
    print(df.to_string(index=False))
    
    # Example queries
    print("\n" + "="*80)
    print(" "*25 + "EXAMPLE QUERIES")
    print("="*80)
    
    # Query 1
    print("\n[Query 1] How do I find my routing number?")
    results = pipeline.query("How do I find my routing number?", n_results=3)
    pipeline.db_manager.print_query_results(results)