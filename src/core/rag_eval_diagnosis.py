"""
Test script to diagnose Milvus setup and data ingestion
"""

from src.core.base import ConfigLoader
from src.core.milvus_manager import MilvusManager
from src.utils.embedding_generator import EmbeddingGenerator
import src.utils.logger as logger

log = logger.Logger()


def test_milvus_setup():
    """Test Milvus connection and data availability"""
    
    log.log_pipeline("Testing Milvus Setup")
    
    # Load configuration
    config_loader = ConfigLoader(config_path="configs/data_processing.yaml")
    ingestion_config = config_loader.configs['ingestion']
    
    try:
        # Test 1: Connect to Milvus
        log.log_step("Test 1: Connecting to Milvus")
        milvus = MilvusManager(
            db_config=ingestion_config['vector_db'],
            embedding_dim=ingestion_config['embeddings']['embedding_dim']
        )
        log.info("✓ Connected successfully")
        
        # Test 2: Check collection statistics
        log.log_step("Test 2: Checking Collection Statistics")
        stats = milvus.get_statistics()
        log.log_messages([
            f"Collection Name: {stats['collection_name']}",
            f"Total Documents: {stats['total_entities']}",
            f"Embedding Dimension: {stats['embedding_dim']}"
        ])
        
        if stats['total_entities'] == 0:
            log.error("❌ PROBLEM: Collection is EMPTY!")
            log.error("You need to run data ingestion first:")
            log.error("  python3 src/core/ingestion.py")
            return False
        else:
            log.info(f"✓ Collection has {stats['total_entities']} documents")
        
        # Test 3: Initialize embedding generator
        log.log_step("Test 3: Testing Embedding Generator")
        embedding_gen = EmbeddingGenerator(
            model_name=ingestion_config['embeddings']['model_name']
        )
        
        test_text = "test query"
        test_embedding = embedding_gen.generate_embedding(test_text)
        log.info(f"✓ Generated embedding with dimension: {len(test_embedding)}")
        
        # Test 4: Try a simple search
        log.log_step("Test 4: Testing Search Functionality")
        query = "How do I find my routing number?"
        log.info(f"Query: '{query}'")
        
        query_embedding = embedding_gen.generate_embedding(query)
        results = milvus.search(query_embedding, top_k=3)
        
        log.info(f"✓ Search returned {len(results)} results")
        
        if len(results) > 0:
            log.info("\nTop Result:")
            top_result = results[0]
            log.log_messages([
                f"  Title: {top_result.get('title', 'N/A')}",
                f"  URL: {top_result.get('url', 'N/A')}",
                f"  Score: {top_result.get('similarity', 0):.4f}",
                f"  Text Preview: {top_result.get('text', '')[:200]}..."
            ])
            log.info("\n✓ All tests passed!")
            return True
        else:
            log.error("❌ PROBLEM: Search returned 0 results")
            log.error("Possible issues:")
            log.error("  1. Data was not properly ingested")
            log.error("  2. Embeddings don't match query embeddings")
            log.error("  3. Collection index is not loaded")
            return False
            
    except Exception as e:
        log.error(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collection_contents():
    """Query and display sample documents from collection"""
    
    log.log_step("Inspecting Collection Contents")
    
    config_loader = ConfigLoader(config_path="configs/data_processing.yaml")
    ingestion_config = config_loader.configs['ingestion']
    
    try:
        milvus = MilvusManager(
            db_config=ingestion_config['vector_db'],
            embedding_dim=ingestion_config['embeddings']['embedding_dim']
        )
        
        # Query first 5 documents
        results = milvus.collection.query(
            expr="id != ''",
            output_fields=["id", "title", "url", "text", "content_type"],
            limit=5
        )
        
        log.info(f"Sample Documents in Collection:")
        for idx, doc in enumerate(results, 1):
            log.log_messages([
                f"\n[Document {idx}]",
                f"  ID: {doc.get('id', 'N/A')}",
                f"  Title: {doc.get('title', 'N/A')}",
                f"  URL: {doc.get('url', 'N/A')}",
                f"  Type: {doc.get('content_type', 'N/A')}",
                f"  Text Preview: {doc.get('text', '')[:150]}..."
            ])
        
    except Exception as e:
        log.error(f"Error querying collection: {e}")


if __name__ == "__main__":
    success = test_milvus_setup()
    
    if success:
        log.info("\n" + "="*80)
        test_collection_contents()
    else:
        log.error("\n" + "="*80)
        log.error("Please fix the issues above before running evaluation")
        log.error("\nSteps to fix:")
        log.error("1. Make sure Milvus is running: docker-compose up -d")
        log.error("2. Run web crawler: python3 src/core/web_crawler_process.py")
        log.error("3. Process PDFs: python3 src/core/web_pdf_preprocessor.py")
        log.error("4. Ingest data: python3 src/core/ingestion.py")
        log.error("5. Run this test again: python3 test_milvus_setup.py")