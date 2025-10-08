"""
RAG System Evaluation Class with Ground Truth Support
Evaluates retrieval quality using various metrics with live vector DB queries
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import src.utils.logger as logger
log = logger.Logger()


class RAGEvaluator:
    """Evaluate RAG system performance with various metrics using live queries"""
    
    def __init__(self, milvus_manager, embedding_generator):
        """
        Initialize evaluator with access to vector DB and embeddings
        
        Args:
            milvus_manager: MilvusManager instance for queries
            embedding_generator: EmbeddingGenerator instance for query embeddings
        """
        self.milvus = milvus_manager
        self.embedding_gen = embedding_generator
        self.evaluation_results = []
        self.ground_truth = {}
    
    def set_ground_truth(self, ground_truth: Dict[str, List[str]]):
        """
        Set ground truth data for evaluation
        
        Args:
            ground_truth: Dict mapping queries to lists of relevant document IDs/URLs
            
        Example:
            ground_truth = {
                "How do I find my routing number?": [
                    "https://www.wellsfargo.com/help/routing-number",
                    "doc_id_123"
                ],
                "What are ATM locations?": [
                    "https://www.wellsfargo.com/atm-locations"
                ]
            }
        """
        self.ground_truth = ground_truth
        log.info(f"✓ Ground truth set for {len(ground_truth)} queries")
    
    def build_ground_truth_interactive(self, queries: List[str], top_k: int = 10) -> Dict[str, List[str]]:
        """
        Helper method to build ground truth by showing retrieved documents
        and allowing manual annotation
        
        Args:
            queries: List of queries to build ground truth for
            top_k: Number of documents to show per query
            
        Returns:
            Dictionary of ground truth data
        """
        ground_truth = {}
        
        log.log_pipeline("Building Ground Truth - Interactive Mode")
        
        for query in queries:
            log.info(f"\n{'='*80}")
            log.info(f"Query: '{query}'")
            log.info(f"{'='*80}")
            
            # Get results
            query_embedding = self.embedding_gen.generate_embedding(query)
            results = self.milvus.search(query_embedding, top_k=top_k)
            
            # Display results
            log.info(f"\nTop {len(results)} retrieved documents:")
            for idx, doc in enumerate(results, 1):
                log.log_messages([
                    f"\n[{idx}] Score: {doc['similarity']:.4f}",
                    f"    ID: {doc.get('id', 'N/A')}",
                    f"    URL: {doc.get('url', 'N/A')}",
                    f"    Title: {doc.get('title', 'N/A')}",
                    f"    Text: {doc.get('text', '')[:200]}..."
                ])
            
            # Collect relevant IDs (in practice, you'd do this manually)
            log.info("\n👆 Review these results and identify relevant document IDs/URLs")
            log.info("For this example, storing all IDs for demonstration...")
            
            # Auto-select documents above a threshold (you can modify this logic)
            relevant_ids = [
                doc.get('id', doc.get('url')) 
                for doc in results 
                if doc.get('similarity', 0) > 0.5  # Adjust threshold as needed
            ]
            
            ground_truth[query] = relevant_ids
            log.info(f"✓ Marked {len(relevant_ids)} documents as relevant")
        
        self.ground_truth = ground_truth
        return ground_truth
    
    def save_ground_truth(self, filepath: str):
        """Save ground truth to JSON file"""
        output_file = Path(filepath)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.ground_truth, f, indent=2, ensure_ascii=False)
        
        log.info(f"✓ Ground truth saved to: {filepath}")
    
    def load_ground_truth(self, filepath: str):
        """Load ground truth from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
        
        log.info(f"✓ Ground truth loaded for {len(self.ground_truth)} queries from: {filepath}")
    
    def evaluate_query(self, 
                      query: str,
                      top_k: int = 5,
                      relevant_doc_ids: Optional[List[str]] = None,
                      search_method: str = 'semantic') -> Dict[str, Any]:
        """
        Evaluate a single query by retrieving from vector DB
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            relevant_doc_ids: List of IDs of truly relevant documents (for metrics)
            search_method: 'semantic', 'keyword', or 'hybrid'
            
        Returns:
            Dictionary with evaluation metrics and retrieved results
        """
        log.log_step(f"Evaluating query: '{query}' (method: {search_method})")
        
        # Use ground truth if available and not provided
        if relevant_doc_ids is None and query in self.ground_truth:
            relevant_doc_ids = self.ground_truth[query]
            log.info(f"Using ground truth: {len(relevant_doc_ids)} relevant documents")
        
        # Check collection status
        try:
            stats = self.milvus.get_statistics()
            log.info(f"Collection: {stats['total_entities']} documents in '{stats['collection_name']}'")
            if stats['total_entities'] == 0:
                log.error("WARNING: Collection is empty!")
                return {
                    'query': query,
                    'search_method': search_method,
                    'num_retrieved': 0,
                    'retrieved_docs': [],
                    'retrieved_doc_ids': [],
                    'error': 'Empty collection'
                }
        except Exception as e:
            log.error(f"Error checking collection: {e}")
        
        # Get query embedding
        try:
            query_embedding = self.embedding_gen.generate_embedding(query)
        except Exception as e:
            log.error(f"Error generating embedding: {e}")
            return {
                'query': query,
                'search_method': search_method,
                'num_retrieved': 0,
                'retrieved_docs': [],
                'retrieved_doc_ids': [],
                'error': f'Embedding generation failed: {e}'
            }
        
        # Retrieve results based on method
        try:
            if search_method == 'semantic':
                retrieved_docs = self.milvus.search(query_embedding, top_k=top_k)
            elif search_method == 'keyword':
                retrieved_docs = self.milvus.keyword_search(query, top_k=top_k)
            elif search_method == 'hybrid':
                retrieved_docs = self.milvus.hybrid_search(query, query_embedding, top_k=top_k)
            else:
                raise ValueError(f"Unknown search method: {search_method}")
            
            log.info(f"Search returned {len(retrieved_docs)} results")
        except Exception as e:
            log.error(f"Error during search: {e}")
            return {
                'query': query,
                'search_method': search_method,
                'num_retrieved': 0,
                'retrieved_docs': [],
                'retrieved_doc_ids': [],
                'error': f'Search failed: {e}'
            }
        
        metrics = {
            'query': query,
            'search_method': search_method,
            'num_retrieved': len(retrieved_docs),
            'retrieved_docs': retrieved_docs,
            'retrieved_doc_ids': [doc.get('id', doc.get('url')) for doc in retrieved_docs],
            'has_ground_truth': relevant_doc_ids is not None
        }
        
        # If we have ground truth, calculate precision/recall metrics
        if relevant_doc_ids:
            metrics.update(self._calculate_retrieval_metrics(retrieved_docs, relevant_doc_ids))
        
        # Score distribution (always calculated)
        if retrieved_docs:
            scores = [doc.get('similarity', doc.get('combined_score', 0)) for doc in retrieved_docs]
            metrics['score_stats'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
            
            # Ranking quality
            metrics['ranking_quality'] = self.evaluate_ranking_quality(retrieved_docs)
        
        self.evaluation_results.append(metrics)
        return metrics
    
    def evaluate_multiple_queries(self, 
                                 queries: List[str], 
                                 top_k: int = 5,
                                 search_method: str = 'semantic',
                                 ground_truth: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
        """
        Evaluate multiple queries
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            search_method: 'semantic', 'keyword', or 'hybrid'
            ground_truth: Optional dict mapping queries to lists of relevant doc IDs
            
        Returns:
            List of evaluation results for each query
        """
        log.log_pipeline(f"Evaluating {len(queries)} queries using {search_method} search")
        
        # Update ground truth if provided
        if ground_truth:
            self.set_ground_truth(ground_truth)
        
        results = []
        for idx, query in enumerate(queries, 1):
            log.info(f"\n[Query {idx}/{len(queries)}]")
            
            relevant_ids = self.ground_truth.get(query) if self.ground_truth else None
            metrics = self.evaluate_query(query, top_k, relevant_ids, search_method)
            results.append(metrics)
            
            # Print results for this query
            self._print_query_results(metrics)
        
        return results
    
    def compare_search_methods(self, 
                              queries: List[str],
                              top_k: int = 5,
                              ground_truth: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Compare semantic, keyword, and hybrid search for multiple queries
        
        Args:
            queries: List of queries to test
            top_k: Number of results per query
            ground_truth: Optional dict of query -> relevant doc IDs
            
        Returns:
            Comparison results
        """
        log.log_pipeline("Comparing Search Methods: Semantic vs Keyword vs Hybrid")
        
        # Update ground truth if provided
        if ground_truth:
            self.set_ground_truth(ground_truth)
        
        comparisons = []
        
        for query in queries:
            log.info(f"\n{'='*80}")
            log.info(f"Query: '{query}'")
            log.info(f"{'='*80}")
            
            query_embedding = self.embedding_gen.generate_embedding(query)
            relevant_ids = self.ground_truth.get(query) if self.ground_truth else None
            
            if relevant_ids:
                log.info(f"Ground truth: {len(relevant_ids)} relevant documents")
            else:
                log.info("No ground truth available - metrics will be limited")
            
            # Get results from all three methods
            semantic_results = self.milvus.search(query_embedding, top_k=top_k)
            keyword_results = self.milvus.keyword_search(query, top_k=top_k)
            hybrid_results = self.milvus.hybrid_search(query, query_embedding, top_k=top_k)
            
            comparison = {
                'query': query,
                'has_ground_truth': relevant_ids is not None,
                'semantic': self._evaluate_method(semantic_results, relevant_ids, 'Semantic'),
                'keyword': self._evaluate_method(keyword_results, relevant_ids, 'Keyword'),
                'hybrid': self._evaluate_method(hybrid_results, relevant_ids, 'Hybrid')
            }
            
            # Determine best method if ground truth available
            if relevant_ids:
                f1_scores = {
                    'semantic': comparison['semantic']['metrics'].get('f1_score', 0),
                    'keyword': comparison['keyword']['metrics'].get('f1_score', 0),
                    'hybrid': comparison['hybrid']['metrics'].get('f1_score', 0)
                }
                comparison['best_method'] = max(f1_scores, key=f1_scores.get)
                comparison['f1_scores'] = f1_scores
                
                log.info(f"\nBest Method: {comparison['best_method'].upper()} (F1: {f1_scores[comparison['best_method']]:.4f})")
            
            comparisons.append(comparison)
        
        # Aggregate results
        summary = self._aggregate_comparison_results(comparisons)
        
        return {
            'individual_comparisons': comparisons,
            'summary': summary
        }
    
    def _print_query_results(self, metrics: Dict[str, Any]):
        """Print results for a single query evaluation"""
        log.log_messages([
            f"Retrieved: {metrics['num_retrieved']} documents",
            f"Search Method: {metrics['search_method']}",
            f"Ground Truth Available: {'Yes' if metrics.get('has_ground_truth') else 'No'}"
        ])
        
        if 'score_stats' in metrics:
            stats = metrics['score_stats']
            log.log_messages([
                f"Score Stats - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}"
            ])
        
        if 'precision' in metrics:
            log.log_messages([
                f"Precision: {metrics['precision']:.4f}",
                f"Recall: {metrics['recall']:.4f}",
                f"F1 Score: {metrics['f1_score']:.4f}",
                f"MAP: {metrics['average_precision']:.4f}",
                f"NDCG: {metrics['ndcg']:.4f}"
            ])
        
        # Print top results
        if metrics['retrieved_docs']:
            log.info("\nTop Results:")
            for idx, doc in enumerate(metrics['retrieved_docs'][:3], 1):
                score = doc.get('similarity', doc.get('combined_score', 0))
                log.log_messages([
                    f"  [{idx}] Score: {score:.4f}",
                    f"      Title: {doc.get('title', 'N/A')}",
                    f"      URL: {doc.get('url', 'N/A')}",
                    f"      Preview: {doc.get('text', '')[:150]}..."
                ])
    
    def _calculate_retrieval_metrics(self, retrieved_docs: List[Dict], relevant_ids: List[str]) -> Dict:
        """Calculate precision, recall, F1, and other retrieval metrics"""
        retrieved_ids = [doc.get('id', doc.get('url')) for doc in retrieved_docs]
        
        # True positives: relevant docs that were retrieved
        tp = len(set(retrieved_ids) & set(relevant_ids))
        
        # False positives: irrelevant docs that were retrieved
        fp = len(set(retrieved_ids) - set(relevant_ids))
        
        # False negatives: relevant docs that were not retrieved
        fn = len(set(relevant_ids) - set(retrieved_ids))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Mean Average Precision (MAP)
        average_precision = self._calculate_average_precision(retrieved_ids, relevant_ids)
        
        # Normalized Discounted Cumulative Gain (NDCG)
        ndcg = self._calculate_ndcg(retrieved_ids, relevant_ids)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'average_precision': average_precision,
            'ndcg': ndcg
        }
    
    def _calculate_average_precision(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate Average Precision"""
        if not relevant_ids:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(relevant_ids) if relevant_ids else 0.0
    
    def _calculate_ndcg(self, retrieved_ids: List[str], relevant_ids: List[str], k: Optional[int] = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        if k is None:
            k = len(retrieved_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_ranking_quality(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Evaluate the quality of ranking based on score distribution"""
        if not retrieved_docs:
            return {'ranking_quality': 'no_results'}
        
        scores = [doc.get('similarity', doc.get('combined_score', 0)) for doc in retrieved_docs]
        
        # Check score separation
        score_range = max(scores) - min(scores)
        score_variance = np.var(scores)
        
        # Good ranking should have clear separation
        quality = 'excellent' if score_range > 0.3 and score_variance > 0.01 else \
                 'good' if score_range > 0.2 else \
                 'poor'
        
        return {
            'ranking_quality': quality,
            'score_range': float(score_range),
            'score_variance': float(score_variance),
            'top_score': float(max(scores)),
            'score_distribution': self._get_score_distribution(scores)
        }
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of scores in bins"""
        bins = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
        for score in scores:
            if score < 0.2:
                bins['0.0-0.2'] += 1
            elif score < 0.4:
                bins['0.2-0.4'] += 1
            elif score < 0.6:
                bins['0.4-0.6'] += 1
            elif score < 0.8:
                bins['0.6-0.8'] += 1
            else:
                bins['0.8-1.0'] += 1
        return bins
    
    def _evaluate_method(self, results: List[Dict], relevant_ids: Optional[List[str]], method_name: str) -> Dict:
        """Evaluate a single search method"""
        log.info(f"\n{method_name} Search Results:")
        
        evaluation = {
            'num_results': len(results),
            'ranking_quality': self.evaluate_ranking_quality(results)
        }
        
        if relevant_ids:
            evaluation['metrics'] = self._calculate_retrieval_metrics(results, relevant_ids)
            metrics = evaluation['metrics']
            log.log_messages([
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall: {metrics['recall']:.4f}",
                f"  F1 Score: {metrics['f1_score']:.4f}"
            ])
        else:
            log.info("  (No ground truth - metrics unavailable)")
        
        # Print top result
        if results:
            top_doc = results[0]
            score = top_doc.get('similarity', top_doc.get('combined_score', 0))
            log.info(f"  Top Result: {top_doc.get('title', 'N/A')} (score: {score:.4f})")
        
        return evaluation
    
    def _aggregate_comparison_results(self, comparisons: List[Dict]) -> Dict:
        """Aggregate comparison results across all queries"""
        if not comparisons:
            return {}
        
        has_ground_truth = any(c.get('has_ground_truth', False) for c in comparisons)
        
        summary = {
            'total_queries': len(comparisons),
            'queries_with_ground_truth': sum(1 for c in comparisons if c.get('has_ground_truth', False)),
            'methods': {}
        }
        
        for method in ['semantic', 'keyword', 'hybrid']:
            method_results = [c[method] for c in comparisons]
            
            # Check if we have metrics
            results_with_metrics = [r for r in method_results if 'metrics' in r]
            
            if results_with_metrics:
                summary['methods'][method] = {
                    'avg_precision': float(np.mean([r['metrics']['precision'] for r in results_with_metrics])),
                    'avg_recall': float(np.mean([r['metrics']['recall'] for r in results_with_metrics])),
                    'avg_f1': float(np.mean([r['metrics']['f1_score'] for r in results_with_metrics])),
                    'avg_map': float(np.mean([r['metrics']['average_precision'] for r in results_with_metrics])),
                    'avg_ndcg': float(np.mean([r['metrics']['ndcg'] for r in results_with_metrics]))
                }
            else:
                summary['methods'][method] = {
                    'avg_num_results': float(np.mean([r['num_results'] for r in method_results])),
                    'note': 'No ground truth available for metric calculation'
                }
        
        return summary
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all evaluated queries"""
        if not self.evaluation_results:
            return {'message': 'No evaluations performed yet'}
        
        # Count queries with ground truth
        queries_with_ground_truth = sum(1 for r in self.evaluation_results if r.get('has_ground_truth', False))
        
        # Aggregate metrics only from queries with ground truth
        metrics_with_values = [r for r in self.evaluation_results if 'precision' in r]
        
        if not metrics_with_values:
            return {
                'total_queries': len(self.evaluation_results),
                'queries_with_ground_truth': queries_with_ground_truth,
                'message': 'No ground truth provided - precision/recall metrics unavailable',
                'note': 'Ranking quality and score distribution are still available in individual results'
            }
        
        summary = {
            'total_queries': len(self.evaluation_results),
            'queries_with_ground_truth': queries_with_ground_truth,
            'avg_precision': float(np.mean([r['precision'] for r in metrics_with_values])),
            'avg_recall': float(np.mean([r['recall'] for r in metrics_with_values])),
            'avg_f1': float(np.mean([r['f1_score'] for r in metrics_with_values])),
            'avg_map': float(np.mean([r['average_precision'] for r in metrics_with_values])),
            'avg_ndcg': float(np.mean([r['ndcg'] for r in metrics_with_values])),
        }
        
        return summary
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results = {
            'ground_truth': self.ground_truth,
            'individual_results': convert_to_serializable(self.evaluation_results),
            'summary': convert_to_serializable(self.get_summary_statistics())
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"✓ Evaluation results saved to: {output_path}")
    
    def print_summary(self):
        """Print evaluation summary"""
        summary = self.get_summary_statistics()
        
        log.log_step("Evaluation Summary")
        
        if 'message' in summary:
            log.info(f"Total Queries Evaluated: {summary.get('total_queries', 0)}")
            log.info(f"Queries with Ground Truth: {summary.get('queries_with_ground_truth', 0)}")
            log.info(f"\n{summary['message']}")
            if 'note' in summary:
                log.info(f"{summary['note']}")
        else:
            log.log_messages([
                f"Total Queries Evaluated: {summary.get('total_queries', 0)}",
                f"Queries with Ground Truth: {summary.get('queries_with_ground_truth', 0)}",
                f"Average Precision: {summary.get('avg_precision', 0):.4f}",
                f"Average Recall: {summary.get('avg_recall', 0):.4f}",
                f"Average F1 Score: {summary.get('avg_f1', 0):.4f}",
                f"Average MAP: {summary.get('avg_map', 0):.4f}",
                f"Average NDCG: {summary.get('avg_ndcg', 0):.4f}"
            ])


# Example usage with test queries and ground truth
if __name__ == "__main__":
    from src.core.milvus_manager import MilvusManager
    from src.utils.embedding_generator import EmbeddingGenerator
    from src.core.base import ConfigLoader
    
    # Initialize
    log.log_pipeline("Starting RAG Evaluation with Ground Truth")
    config_loader = ConfigLoader(config_path="configs/data_processing.yaml")
    ingestion_config = config_loader.configs['ingestion']
    
    # Initialize Milvus and Embeddings
    milvus = MilvusManager(
        db_config=ingestion_config['vector_db'],
        embedding_dim=ingestion_config['embeddings']['embedding_dim'],
        drop_existing=False  # Don't drop existing data
    )
    
    embedding_gen = EmbeddingGenerator(
        model_name=ingestion_config['embeddings']['model_name']
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator(milvus, embedding_gen)
    
    # Test queries
    test_queries = [
        "How do I find my routing number?",
        "What are the ATM locations?",
        "How do I use Zelle?"
    ]
    
    # Option 1: Build ground truth interactively
    log.log_step("Building Ground Truth")
    ground_truth = evaluator.build_ground_truth_interactive(test_queries, top_k=10)
    evaluator.save_ground_truth("evaluation_results/ground_truth.json")
    

    # Ground truth - URLs from your actual data
    ground_truth = {
        "How do I find my routing number?": [
            "Preview: Learn more What’s a routing number and where do you find it"
        ],
        "What are the ATM locations?": [
            "Preview: location near you. ZIP code to find a branch Find an ATM or banking location near you"
        ],
        "How do I use Zelle?": [
            "Zelle ® ? Zelle ® is a convenient way to send money"
        ]
    }
    evaluator.set_ground_truth(ground_truth)
    
    # Evaluate with semantic search
    log.log_step("Testing Semantic Search with Ground Truth")
    semantic_results = evaluator.evaluate_multiple_queries(
        queries=test_queries,
        top_k=5,
        search_method='semantic'
    )
    
    # Compare all search methods
    log.log_step("Comparing Search Methods")
    comparison = evaluator.compare_search_methods(
        queries=test_queries,
        top_k=5
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results("evaluation_results/test_results_with_ground_truth.json")
    
    log.log_pipeline("Evaluation Complete")