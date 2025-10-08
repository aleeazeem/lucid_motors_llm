"""
Configuration Loader
Loads and manages application configuration from YAML file
"""

from src.utils.loader_utils import load_yaml
from typing import Dict, Any
from src.utils.logger import Logger
log = Logger()


class ConfigLoader:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_path: str = "configs/data_processing.yaml"):
        """
        Initialize ConfigLoader
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config_path = config_path
        self.configs = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load and return configurations from YAML file
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            ValueError: If configuration fails to load
        """
        log.info(f"Loading configuration from: {self.config_path}")
        configs = load_yaml(self.config_path)
        if not configs:
            raise ValueError(f"Failed to load configurations from {self.config_path}")
        log.info("✓ Configuration loaded successfully")
        return configs
    
    # General configs
    def get_name(self) -> str:
        """Get application name"""
        return self.configs['configs']['name']
    
    def get_base_url(self) -> str:
        """Get base URL"""
        return self.configs['configs']['base_url']
    
    def get_paths(self) -> str:
        """Get path"""
        return self.configs['configs']['paths']
    
    def get_root_url(self) -> str:
        """Get complete root URL (base_url + paths)"""
        return self.configs['configs']['base_url'] + self.configs['configs']['paths']
    
    def get_output_path(self) -> str:
        """Get base output path"""
        return self.configs['configs']['output_path']
   
    def get_unprocessed_pdf_path(self) -> str:
        """Get unprocessed PDF file storage path"""
        return self.configs['configs']['unprocessed_pdf_file_path']
    
    def get_processed_pdf_path(self) -> str:
        """Get processed PDF file path"""
        return self.configs['configs']['processed_pdf_file_path']
   
    def get_processed_web_path(self) -> str:
        """Get processed web files path"""
        return self.configs['configs']['processed_web_file_path']
    
    # Crawler settings
    def get_crawler_settings(self) -> Dict[str, Any]:
        """Get all crawler configuration settings"""
        return self.configs['crawler']
    
    def get_crawler_timeout(self) -> int:
        """Get crawler timeout"""
        return self.configs['crawler']['timeout']
    
    def get_crawler_max_retries(self) -> int:
        """Get crawler max retries"""
        return self.configs['crawler']['max_retries']
    
    def get_crawler_max_workers(self) -> int:
        """Get crawler max workers"""
        return self.configs['crawler']['max_workers']
    
    def get_crawler_headers(self) -> Dict[str, str]:
        """Get crawler headers"""
        return self.configs['crawler']['headers']
    
    def get_crawler_user_agent(self) -> str:
        """Get crawler user agent"""
        return self.configs['crawler']['user_agent']
    
    def get_max_depth(self) -> int:
        """Get maximum crawl depth"""
        return self.configs['crawler']['max_depth']
    
    def get_max_pages(self) -> int:
        """Get maximum pages to crawl"""
        return self.configs['crawler']['max_pages']
    
    def get_rate_limit(self) -> int:
        """Get rate limit (seconds between requests)"""
        return self.configs['crawler']['rate_limit']
    
    def get_crawl_delay(self) -> int:
        """Get crawl delay"""
        return self.configs['crawler']['crawl_delay']
    
    def get_pdf_download_enabled(self) -> bool:
        """Check if PDF download is enabled"""
        return self.configs['crawler']['pdf_download']
    
    # Ingestion settings
    def get_ingestion_settings(self) -> Dict[str, Any]:
        """Get all ingestion configuration settings"""
        return self.configs['ingestion']
    
    def get_chunk_size(self) -> int:
        """Get chunk size for text chunking"""
        return self.configs['ingestion']['chunk_size']
    
    def get_chunk_overlap(self) -> int:
        """Get chunk overlap percentage"""
        return self.configs['ingestion']['chunk_overlap']
    
    def get_description_generation_enabled(self) -> bool:
        """Check if description generation is enabled"""
        return self.configs['ingestion']['description_generation']
    
    def get_description_model(self) -> str:
        """Get model for description generation"""
        return self.configs['ingestion']['model_description_generation']
    
    # Vector DB settings
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get all vector database configuration settings"""
        return self.configs['ingestion']['vector_db']
    
    def get_vector_db_type(self) -> str:
        """Get vector database type"""
        return self.configs['ingestion']['vector_db']['db_type']
    
    def get_collection_name(self) -> str:
        """Get vector database collection name"""
        return self.configs['ingestion']['vector_db']['collection_name']
    
    def get_vector_db_host(self) -> str:
        """Get vector database host"""
        return self.configs['ingestion']['vector_db']['host']
    
    def get_vector_db_port(self) -> str:
        """Get vector database port"""
        return self.configs['ingestion']['vector_db']['port']
    
    def get_index_type(self) -> str:
        """Get vector database index type"""
        return self.configs['ingestion']['vector_db']['index_type']
    
    # Embeddings settings
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get all embeddings configuration settings"""
        return self.configs['ingestion']['embeddings']
    
    def get_embedding_model_name(self) -> str:
        """Get embedding model name"""
        return self.configs['ingestion']['embeddings']['model_name']
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.configs['ingestion']['embeddings']['embedding_dim']
    
    def get_embedding_batch_size(self) -> int:
        """Get embedding batch size"""
        return self.configs['ingestion']['embeddings']['batch_size']
    
    def get_use_multithreading(self) -> bool:
        """Check if multithreading is enabled for embeddings"""
        return self.configs['ingestion']['embeddings']['use_multithreading']
    
    def get_embedding_num_workers(self) -> int:
        """Get number of workers for embedding generation"""
        return self.configs['ingestion']['embeddings']['num_workers']
    
    def get_show_progress(self) -> bool:
        """Check if progress bar should be shown"""
        return self.configs['ingestion']['embeddings']['show_progress']