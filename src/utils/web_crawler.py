"""
Web Crawler for RAG Projects
Crawls websites up to a specified depth and extracts text from HTML and PDF files
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import threading
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.web_crawler_utils import CrawlerUtils, FileUtils, StatisticsUtils


class WebCrawler:
    """
    Main web crawler class for extracting text from websites
    
    Args:
        start_url (str): The starting URL to crawl from
        max_depth (int): Maximum depth to crawl (default: 3)
        max_workers (int): Number of concurrent workers (default: 10)
        output_base_dir (str): Base output directory (default: 'output')
    """
    
    def __init__(self, start_url, max_depth=3, max_workers=10, output_base_dir='output'):
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_workers = max_workers
        self.visited_urls = set()
        self.crawled_data = []
        self.domain = urlparse(start_url).netloc
        self.lock = threading.Lock()
        
        # Create output directory structure
        # output/domain_name/
        # output/domain_name/pdfs/
        domain_folder = self.domain.replace('.', '_')
        self.output_dir = os.path.join(output_base_dir, domain_folder)
        self.pdf_output_dir = os.path.join(self.output_dir, 'pdfs')
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.pdf_output_dir, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"PDF directory: {self.pdf_output_dir}")
        
        # Initialize utility classes
        self.crawler_utils = CrawlerUtils()
        self.file_utils = FileUtils()
        self.stats_utils = StatisticsUtils()
    
    def sanitize_filename(self, filename):
        """
        Sanitize filename to remove invalid characters
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200] + ext
        return filename
    
    def is_valid_url(self, url):
        """
        Check if URL belongs to the same domain and hasn't been visited
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        parsed = urlparse(url)
        with self.lock:
            is_valid = (parsed.netloc == self.domain and 
                       url not in self.visited_urls and
                       parsed.scheme in ['http', 'https'])
        return is_valid
    
    def mark_visited(self, url):
        """
        Thread-safe way to mark URL as visited
        
        Args:
            url (str): URL to mark as visited
        """
        with self.lock:
            self.visited_urls.add(url)
    
    def is_visited(self, url):
        """
        Thread-safe way to check if URL has been visited
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if URL has been visited, False otherwise
        """
        with self.lock:
            return url in self.visited_urls
    
    def crawl_page(self, url, depth, parent_url):
        """
        Crawl a single page and extract its content and links
        
        Args:
            url (str): URL to crawl
            depth (int): Current depth level
            parent_url (str): Parent URL that linked to this page
            
        Returns:
            tuple: (page_data dict, list of links)
        """
        if self.is_visited(url) or depth > self.max_depth:
            return None, []
        
        self.mark_visited(url)
        print(f"Crawling (depth {depth}): {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Check if it's a PDF
            if (self.crawler_utils.is_pdf(url) or 
                'application/pdf' in response.headers.get('Content-Type', '')):
                print(f"  Processing PDF...")
                
                # Extract filename from URL
                pdf_filename = url.split('/')[-1]
                if not pdf_filename.endswith('.pdf'):
                    pdf_filename = pdf_filename + '.pdf'
                pdf_filename = self.sanitize_filename(pdf_filename)
                
                # Create base name without extension for metadata
                base_name = os.path.splitext(pdf_filename)[0]
                
                # Save PDF file
                pdf_path = os.path.join(self.pdf_output_dir, pdf_filename)
                self.file_utils.save_pdf(response.content, pdf_path)
                
                # Extract text from PDF
                text = self.crawler_utils.extract_text_from_pdf(response.content)
                
                # Create metadata
                page_data = {
                    'url': url,
                    'depth': depth,
                    'parent_url': parent_url,
                    'title': base_name,
                    'description': 'PDF Document',
                    'text_length': len(text),
                    'content_type': 'pdf',
                    'pdf_file': pdf_filename
                }
                
                # Save metadata to JSON file with same name as PDF
                metadata_path = os.path.join(self.pdf_output_dir, f"{base_name}.json")
                self.file_utils.save_pdf_metadata(page_data, metadata_path)
                
                # Add reference to main crawled data (without full text to keep it lightweight)
                crawled_reference = {
                    'url': url,
                    'depth': depth,
                    'parent_url': parent_url,
                    'title': base_name,
                    'description': 'PDF Document',
                    'text_length': len(text),
                    'content_type': 'pdf',
                    'pdf_file': pdf_filename,
                    'metadata_file': f"{base_name}.json"
                }
                
                # PDFs don't have links to follow
                return crawled_reference, []
            
            # Process HTML page
            soup = BeautifulSoup(response.content, 'html.parser')
            text = self.crawler_utils.extract_text_from_html(soup)
            title_text = self.crawler_utils.extract_title_from_html(soup)
            description = self.crawler_utils.extract_description_from_html(soup)
            
            page_data = {
                'url': url,
                'depth': depth,
                'parent_url': parent_url,
                'title': title_text,
                'description': description,
                'text': text,
                'text_length': len(text),
                'content_type': 'html'
            }
            
            # Extract links for next depth
            links = []
            if depth < self.max_depth:
                with self.lock:
                    visited_copy = self.visited_urls.copy()
                links = list(self.crawler_utils.extract_links_from_html(
                    soup, url, self.domain, visited_copy
                ))
                print(f"  Found {len(links)} new links at depth {depth}")
            
            return page_data, links
            
        except Exception as e:
            print(f"  Error crawling {url}: {str(e)}")
            return None, []
    
    def crawl(self):
        """
        Start the crawling process using BFS with concurrent requests
        
        Returns:
            list: List of crawled page data dictionaries
        """
        # Process each depth level sequentially, but URLs within a level concurrently
        current_level = [(self.start_url, 0, None)]
        
        for depth in range(self.max_depth + 1):
            if not current_level:
                break
            
            print(f"\n{'='*60}")
            print(f"Processing depth {depth} - {len(current_level)} URLs")
            print(f"{'='*60}")
            
            next_level = []
            
            # Process current level concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.crawl_page, url, depth, parent): url 
                    for url, d, parent in current_level if d == depth
                }
                
                for future in as_completed(futures):
                    page_data, links = future.result()
                    
                    if page_data:
                        with self.lock:
                            self.crawled_data.append(page_data)
                        
                        # Add links for next level
                        for link in links:
                            if not self.is_visited(link):
                                next_level.append((link, depth + 1, page_data['url']))
            
            current_level = next_level
            print(f"Completed depth {depth}. Found {len(next_level)} URLs for next level.")
        
        return self.crawled_data
    
    def save_results(self, json_filename=None, txt_filename=None):
        """
        Save crawled data to JSON and text files in the output directory
        
        Args:
            json_filename (str): Custom JSON filename (optional)
            txt_filename (str): Custom text filename (optional)
        """
        # Generate default filenames based on domain if not provided
        if json_filename is None:
            domain_name = self.domain.replace('.', '_')
            json_filename = f"{domain_name}_data.json"
        
        if txt_filename is None:
            domain_name = self.domain.replace('.', '_')
            txt_filename = f"{domain_name}_data.txt"
        
        # Create full paths
        json_path = os.path.join(self.output_dir, json_filename)
        txt_path = os.path.join(self.output_dir, txt_filename)
        
        # Save files
        self.file_utils.save_to_json(self.crawled_data, json_path)
        self.file_utils.save_to_text(self.crawled_data, txt_path)
    
    def print_statistics(self):
        """Print crawling statistics"""
        self.stats_utils.print_statistics(self.crawled_data, self.visited_urls)
    
    def print_sample_pages(self):
        """Print sample of crawled pages by depth"""
        self.stats_utils.print_sample_pages(self.crawled_data, self.max_depth)


# Main execution
if __name__ == "__main__":
    # Configuration
    start_url = "https://www.wellsfargo.com/help/"
    max_depth = 3
    max_workers = 10
    
    # Create crawler instance
    crawler = WebCrawler(start_url, max_depth=max_depth, max_workers=max_workers)
    
    print(f"Starting concurrent crawl from {start_url}")
    print(f"Max depth: {crawler.max_depth}")
    print(f"Max workers: {crawler.max_workers}")
    print(f"Domain: {crawler.domain}")
    
    # Start crawling
    start_time = time.time()
    data = crawler.crawl()
    elapsed_time = time.time() - start_time
    
    # Print results
    crawler.print_statistics()
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    # Save results
    crawler.save_results()
    
    # Print sample pages
    crawler.print_sample_pages()