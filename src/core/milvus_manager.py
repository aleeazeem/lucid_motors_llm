"""
Milvus Database Manager - Complete Version with Hybrid Search
"""

from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from typing import List, Dict, Any, Optional
import uuid
from src.core.base import ConfigLoader
import src.utils.logger as logger
log = logger.Logger()


class MilvusManager:
    """Milvus vector database manager with hybrid search capabilities"""

    def __init__(self, db_config: Dict[str, Any], embedding_dim: int = 384, drop_existing: bool = True):
        self.collection_name = db_config['collection_name']
        self.embedding_dim = embedding_dim
        self.index_type = db_config['index_type']
        self.host = db_config['host']
        self.port = db_config['port']
        
        # Connect to Milvus
        connections.connect("default", host=self.host, port=self.port)
        log.info("✓ Connected to Milvus") 
        
        # Drop existing collection if exists (for clean start)
        if drop_existing:
            try:
                utility.drop_collection(self.collection_name)
                log.info(f"Dropped existing collection: {self.collection_name}")
            except:
                pass
            
            # Create new collection
            log.info(f"Creating new collection: {self.collection_name}")
            self.collection = self._create_collection()
            self._create_index()
        else:
            # Use existing collection
            if utility.has_collection(self.collection_name):
                log.info(f"Using existing collection: {self.collection_name}")
                self.collection = Collection(name=self.collection_name)
            else:
                log.info(f"Creating new collection: {self.collection_name}")
                self.collection = self._create_collection()
                self._create_index()
        
        self.collection.load()
        log.info(f"✓ Milvus ready: {self.collection_name}")
    
    def _create_collection(self):
        """Create collection with schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="depth", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="total_chunks", dtype=DataType.INT64),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
        ]
        schema = CollectionSchema(fields=fields, description="RAG collection")
        return Collection(name=self.collection_name, schema=schema)
    
    def _create_index(self):
        """Create vector index"""
        index_params = {
            "metric_type": "L2", 
            "index_type": self.index_type, 
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        log.info("✓ Created vector index")
    
    def insert_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 100):
        """Insert chunks with embeddings into Milvus"""
        log.info(f"\nInserting {len(chunks)} documents...")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            data = [
                [str(uuid.uuid4()) for _ in batch_chunks],
                batch_embeddings,
                [c['chunk_text'] for c in batch_chunks],
                [c['url'] for c in batch_chunks],
                [c['title'][:500] for c in batch_chunks],
                [c['depth'] for c in batch_chunks],
                [c['chunk_index'] for c in batch_chunks],
                [c['total_chunks'] for c in batch_chunks],
                [c['content_type'] for c in batch_chunks]
            ]
            
            self.collection.insert(data)
            log.info(f"  Batch {i // batch_size + 1} inserted")
        
        self.collection.flush()
        log.info(f"✓ Inserted {len(chunks)} documents")
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_expr: Optional[str] = None):
        """Semantic search using vector similarity"""
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "text", "url", "title", "depth", "chunk_index", "total_chunks", "content_type"]
        )
        
        return self._format_results(results, score_type='semantic')
    
    def keyword_search(self, query_text: str, top_k: int = 5, filter_expr: Optional[str] = None):
        """Keyword search using text matching (fallback implementation)"""
        log.info("Using keyword search with text matching")
        return self._fallback_keyword_search(query_text, top_k, filter_expr)
    
    def _fallback_keyword_search(self, query_text: str, top_k: int, filter_expr: Optional[str] = None):
        """Fallback keyword search using query filter"""
        try:
            # Create filter expression for keyword matching
            keywords = query_text.lower().split()
            
            # Query all documents and filter client-side
            results = self.collection.query(
                expr=filter_expr if filter_expr else "id != ''",
                output_fields=["id", "text", "url", "title", "depth", "chunk_index", "total_chunks", "content_type"],
                limit=1000  # Get enough to filter
            )
            
            # Score based on keyword overlap and TF-IDF-like scoring
            scored_results = []
            for doc in results:
                text_lower = doc['text'].lower()
                title_lower = doc['title'].lower()
                
                # Count keyword matches in text and title (title weighted more)
                text_matches = sum(1 for kw in keywords if kw in text_lower)
                title_matches = sum(1 for kw in keywords if kw in title_lower)
                
                # Calculate score (title matches count double)
                total_matches = text_matches + (title_matches * 2)
                score = total_matches / (len(keywords) * 2)  # Normalize
                
                if score > 0:
                    scored_results.append({
                        'id': doc['id'],
                        'keyword_score': score,
                        'bm25_score': score,
                        'similarity': score,
                        'text': doc['text'],
                        'url': doc['url'],
                        'title': doc['title'],
                        'depth': doc['depth'],
                        'chunk_index': doc['chunk_index'],
                        'total_chunks': doc['total_chunks'],
                        'content_type': doc['content_type']
                    })
            
            # Sort by score and return top_k
            scored_results.sort(key=lambda x: x['keyword_score'], reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            log.error(f"Fallback keyword search failed: {e}")
            return []
    
    def hybrid_search(self, query_text: str, query_embedding: List[float], 
                     top_k: int = 5, alpha: float = 0.7, filter_expr: Optional[str] = None):
        """
        Hybrid search combining semantic and keyword search
        
        Args:
            query_text: Text query for keyword search
            query_embedding: Vector embedding for semantic search
            top_k: Number of results to return
            alpha: Weight for semantic search (0-1). 1.0 = pure semantic, 0.0 = pure keyword
            filter_expr: Optional filter expression
            
        Returns:
            List of combined and re-ranked results
        """
        # Get results from both search methods
        semantic_results = self.search(query_embedding, top_k=top_k * 2, filter_expr=filter_expr)
        keyword_results = self.keyword_search(query_text, top_k=top_k * 2, filter_expr=filter_expr)
        
        # Combine and re-rank
        combined = self._combine_results(semantic_results, keyword_results, alpha)
        
        return combined[:top_k]
    
    def _combine_results(self, semantic_results: List[Dict], keyword_results: List[Dict], alpha: float):
        """Combine semantic and keyword results with weighted scoring"""
        from collections import defaultdict
        
        # Create a dictionary to merge results by ID
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.get('id', result.get('url'))
            combined[doc_id] = result.copy()
            combined[doc_id]['semantic_score'] = result.get('similarity', 0)
            combined[doc_id]['keyword_score'] = 0
        
        # Add/merge keyword results
        for result in keyword_results:
            doc_id = result.get('id', result.get('url'))
            if doc_id in combined:
                combined[doc_id]['keyword_score'] = result.get('keyword_score', result.get('bm25_score', 0))
            else:
                combined[doc_id] = result.copy()
                combined[doc_id]['semantic_score'] = 0
                combined[doc_id]['keyword_score'] = result.get('keyword_score', result.get('bm25_score', 0))
        
        # Calculate combined score
        for doc_id in combined:
            semantic = combined[doc_id]['semantic_score']
            keyword = combined[doc_id]['keyword_score']
            combined[doc_id]['combined_score'] = alpha * semantic + (1 - alpha) * keyword
            combined[doc_id]['similarity'] = combined[doc_id]['combined_score']
        
        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return results
    
    def _format_results(self, results, score_type='semantic'):
        """Format search results"""
        formatted = []
        for hits in results:
            for hit in hits:
                score = 1 / (1 + hit.distance) if hasattr(hit, 'distance') else hit.score
                doc = {
                    'id': hit.entity.get('id') if hasattr(hit, 'entity') else hit.get('id'),
                    'text': hit.entity.get('text') if hasattr(hit, 'entity') else hit.get('text'),
                    'url': hit.entity.get('url') if hasattr(hit, 'entity') else hit.get('url'),
                    'title': hit.entity.get('title') if hasattr(hit, 'entity') else hit.get('title'),
                    'depth': hit.entity.get('depth') if hasattr(hit, 'entity') else hit.get('depth'),
                    'chunk_index': hit.entity.get('chunk_index') if hasattr(hit, 'entity') else hit.get('chunk_index'),
                    'total_chunks': hit.entity.get('total_chunks') if hasattr(hit, 'entity') else hit.get('total_chunks'),
                    'content_type': hit.entity.get('content_type') if hasattr(hit, 'entity') else hit.get('content_type'),
                }
                
                if score_type == 'semantic':
                    doc['similarity'] = score
                    doc['semantic_score'] = score
                    doc['keyword_score'] = 0
                else:
                    doc['keyword_score'] = score
                    doc['bm25_score'] = score
                    doc['semantic_score'] = 0
                    doc['similarity'] = score
                
                formatted.append(doc)
        return formatted
    
    def get_statistics(self):
        """Get collection stats"""
        self.collection.flush()
        return {
            'collection_name': self.collection_name,
            'total_entities': self.collection.num_entities,
            'embedding_dim': self.embedding_dim
        }