"""
Web Crawler Utilities
Contains helper functions for text extraction, file operations, and statistics
"""

import json
import pdfplumber
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class CrawlerUtils:
    """Utility class for text extraction and processing"""
    
    @staticmethod
    def is_pdf(url):
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf')
    
    @staticmethod
    def extract_text_from_html(soup):
        """Extract clean text from HTML soup object"""
        # Remove unwanted HTML elements
        for script in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    @staticmethod
    def extract_text_from_pdf(pdf_content):
        """Extract text from PDF bytes using pdfplumber"""
        try:
            pdf_file = BytesIO(pdf_content)
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"[Page {page_num}]\n{page_text}\n\n"
            return text.strip()
        except Exception as e:
            print(f"    Error extracting PDF text: {str(e)}")
            return ""
    
    @staticmethod
    def extract_links_from_html(soup, current_url, domain, visited_urls):
        """Extract all valid links from the HTML page"""
        links = set()
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(current_url, link['href'])
            absolute_url = absolute_url.split('#')[0].rstrip('/')
            
            parsed = urlparse(absolute_url)
            # Check if it's valid: same domain, not visited, proper scheme
            if (parsed.netloc == domain and 
                absolute_url not in visited_urls and
                parsed.scheme in ['http', 'https']):
                links.add(absolute_url)
        
        return links
    
    @staticmethod
    def extract_title_from_html(soup):
        """Extract title from HTML soup object"""
        title = soup.find('title')
        return title.get_text(strip=True) if title else ''
    
    @staticmethod
    def extract_description_from_html(soup):
        """Extract meta description from HTML soup object"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else ''


class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def save_to_json(data, filename='crawled_data.json'):
        """Save crawled data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nData saved to {filename}")
    
    @staticmethod
    def save_to_text(data, filename='crawled_data.txt'):
        """Save crawled data to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"URL: {item['url']}\n")
                f.write(f"Depth: {item['depth']}\n")
                f.write(f"Parent URL: {item['parent_url']}\n")
                f.write(f"Content Type: {item.get('content_type', 'html')}\n")
                f.write(f"Title: {item['title']}\n")
                f.write(f"Description: {item['description']}\n")
                f.write(f"Text Length: {item['text_length']}\n")
                
                # Check if it's a PDF (no text field)
                if item.get('content_type') == 'pdf':
                    f.write(f"PDF File: {item.get('pdf_file', 'N/A')}\n")
                    f.write(f"Metadata File: {item.get('metadata_file', 'N/A')}\n")
                    f.write("-" * 80 + "\n")
                    f.write("[PDF content saved separately]\n")
                else:
                    # HTML content - write the text
                    f.write("-" * 80 + "\n")
                    f.write(item.get('text', ''))
                
                f.write("\n" + "=" * 80 + "\n\n")
        print(f"Data saved to {filename}")
    
    @staticmethod
    def save_pdf(pdf_content, filename):
        """
        Save PDF content to a file
        
        Args:
            pdf_content (bytes): PDF file content
            filename (str): Output filename for the PDF
        """
        try:
            with open(filename, 'wb') as f:
                f.write(pdf_content)
            print(f"  PDF saved: {filename}")
        except Exception as e:
            print(f"  Error saving PDF {filename}: {str(e)}")
    
    @staticmethod
    def save_pdf_metadata(metadata, filename):
        """
        Save PDF metadata to a JSON file
        
        Args:
            metadata (dict): PDF metadata dictionary
            filename (str): Output filename for the JSON
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"  Metadata saved: {filename}")
        except Exception as e:
            print(f"  Error saving metadata {filename}: {str(e)}")


class StatisticsUtils:
    """Utility class for statistics and reporting"""
    
    @staticmethod
    def print_statistics(crawled_data, visited_urls):
        """Print crawling statistics"""
        print(f"\n{'='*60}")
        print(f"Crawl Complete!")
        print(f"{'='*60}")
        print(f"Total pages crawled: {len(crawled_data)}")
        print(f"Total URLs visited: {len(visited_urls)}")
        
        # Pages by depth
        depth_counts = {}
        for item in crawled_data:
            depth = item['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        print(f"\nPages by depth:")
        for depth in sorted(depth_counts.keys()):
            print(f"  Depth {depth}: {depth_counts[depth]} pages")
        
        # Total text extracted
        total_chars = sum(item['text_length'] for item in crawled_data)
        if crawled_data:
            print(f"\nTotal text extracted: {total_chars:,} characters")
            print(f"Average text per page: {total_chars // len(crawled_data):,} characters")
    
    @staticmethod
    def print_sample_pages(crawled_data, max_depth):
        """Print sample of crawled pages by depth"""
        print(f"\n{'='*60}")
        print("Sample of crawled pages:")
        print(f"{'='*60}")
        for depth in range(max_depth + 1):
            depth_pages = [item for item in crawled_data if item['depth'] == depth]
            if depth_pages:
                print(f"\nDepth {depth} ({len(depth_pages)} pages):")
                for item in depth_pages[:3]:
                    print(f"  - {item['title'][:60]}... ({item['text_length']:,} chars)")
                    print(f"    URL: {item['url']}")