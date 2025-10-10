"""
Ingest JSON files to Milvus
Uses Fixed-size chunking with overlap for optimal RAG performance
"""

import json
import sys

from src.utils.loader_utils import load_all_json_files, load_all_files
from typing import Tuple, Optional

from src.utils.logger import Logger
from src.utils.chunk_utils import TextChunker
from src.utils.embedding_generator import EmbeddingGenerator
from src.core.milvus_manager import MilvusManager
from src.utils.logger import Logger
from src.core.base import ConfigLoader

log = Logger()



def ingest_json_to_milvus(config_loader: ConfigLoader) -> Tuple[Optional[MilvusManager], Optional[EmbeddingGenerator]]:
    """
    Complete ingestion pipeline: Load -> Chunk -> Embed -> Store in Milvus
    
    Args:
        config_loader (ConfigLoader): Configuration loader instance
    
    Returns:
        tuple: (MilvusManager, EmbeddingGenerator) instances
    """
    
    # Get configurations
    log.log_pipeline("STARTING INGESTION PIPELINE")
    ingestion_config = config_loader.get_ingestion_settings()
    
    # STEP 1: Load JSON files
    log.log_step("STEP 1: Loading JSON files")
    all_documents = load_all_files(config_loader.get_processed_pdf_path(), "json")
    if not all_documents:
        log.error("No documents loaded from the provided JSON files.")
        raise ValueError("No documents loaded. Please check the input files.")
        

    log.log_step("STEP 2: Chunking Documents (Fixed-size with Overlap)")
    chunk_size = ingestion_config['chunk_size']
    chunk_overlap = ingestion_config['chunk_overlap']
    log.info(f"\nUsing chunk size of {chunk_size} words with {chunk_overlap}% overlap.")
    
    chunker = TextChunker(
        chunk_size_words=chunk_size,
        overlap_percentage=int((chunk_overlap / chunk_size) * 100)
    )
    chunks = chunker.chunk_documents_batch(all_documents)
    log.info(f"\nTotal chunks created: {len(chunks)}")
    
    
    
    # STEP 3: Generate embeddings
    log.log_step("STEP 3: Generating Embeddings")
    embedding_gen = EmbeddingGenerator(ingestion_config['embeddings']['model_name'])
    
    log.info(f"\nGenerating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_gen.generate_embeddings_for_chunks(
        chunks, 
        batch_size=ingestion_config['embeddings']['batch_size'],
        use_multithreading=ingestion_config['embeddings']['use_multithreading'],
        num_workers=ingestion_config['embeddings']['num_workers'],
        show_progress=ingestion_config['embeddings']['show_progress']
    )
    log.info(f"\nTotal embeddings generated: {len(embeddings)}")
    log.info(f"Embedding dimension: {ingestion_config['embeddings']['embedding_dim']}")
    
    # STEP 4: Store in Milvus
    log.log_step("STEP 4: Storing in Milvus")
    vector_db_config = ingestion_config['vector_db']
    milvus = MilvusManager(db_config=vector_db_config, embedding_dim=ingestion_config['embeddings']['embedding_dim'])
    
    milvus.insert_documents(chunks, embeddings, batch_size=100)
    
    # STEP 5: Verify and show statistics
    log.log_step("STEP 5: Pipeline Summary & Statistics")
    stats = milvus.get_statistics()
    log.info(f"  Input Files: {len(all_documents)}")
    log.info(f"  Documents Processed: {len(all_documents)}")
    log.info(f"  Chunks Created: {len(chunks)}")
    log.info(f"  Embeddings Generated: {len(embeddings)}")
    log.info(f"  Milvus Collection: {stats['collection_name']}")
    log.info(f"  Total Entities in DB: {stats['total_entities']}")
    log.info(f"  Embedding Dimension: {stats['embedding_dim']}")
    
    log.log_pipeline("INGESTION PIPELINE COMPLETED") 
    return milvus, embedding_gen


"""def test_query(milvus: MilvusManager, embedding_gen: EmbeddingGenerator, 
               query_text: str, top_k: int = 3):
    log.log_step(f"Query: '{query_text}'")
    
    query_embedding = embedding_gen.generate_embedding(query_text)
    results = milvus.search(query_embedding, top_k=top_k)
    
    if not results:
        log.error("No results found.")
        return
    
    for idx, result in enumerate(results, 1):
        log.log_messages([f"\n[Result {idx}]",
                            f"Similarity Score: {result['similarity']:.4f}",
                            f"Title: {result['title']}",
                            f"URL: {result['url']}",
                            f"Depth: {result['depth']}",
                            f"Chunk: {result['chunk_index'] + 1}/{result['total_chunks']}",
                            f"Text Preview: {result['text'][:200]}...\n"
                         ])
"""

# Main execution
if __name__ == "__main__":
    config_loader = ConfigLoader(config_path="configs/data_processing.yaml")
    milvus, embedding_gen = ingest_json_to_milvus(config_loader)
    
    # Exit if ingestion failed
    if milvus is None or embedding_gen is None:
        print("\nIngestion failed. Please check errors above.")
        exit(1)
    
    # Test with sample queries
    log.log_pipeline("TESTING INGESTION WITH SAMPLE QUERIES")
    
    """
    test_queries = [
        "How do I find my routing number?",
        "What are the ATM locations?",
        "How do I use Zelle?"
    ]
    
    for query in test_queries:
        test_query(milvus, embedding_gen, query, top_k=3)
        """