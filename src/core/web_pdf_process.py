import json
import os
from pathlib import Path
from typing import Dict, List, Any
import base64
from openai import OpenAI
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml
import sys
from src.core.base import ConfigLoader
from src.utils.logger import Logger

log = Logger()

load_dotenv()


class RAGPreprocessor:
    def __init__(self, openai_api_key: str, input_dir: str, output_dir: str, max_workers: int = 5, image_description_generation: bool = False):
        """
        Initialize the RAG preprocessor.
        
        Args:
            openai_api_key: OpenAI API key for image description
            input_dir: Directory containing PDF and JSON file pairs
            output_dir: Directory to save processed JSON file
            max_workers: Maximum number of parallel threads for image processing
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.image_description_generation = image_description_generation
        self.print_lock = threading.Lock()
        
        self.converter = DocumentConverter()
    
    def thread_safe_print(self, message: str):
        """Thread-safe printing."""
        with self.print_lock:
            log.info(message)
    
    def find_file_pairs(self) -> List[tuple]:
        """Find all matching PDF and JSON file pairs in the input directory."""
        pairs = []
        json_files = list(self.input_dir.glob("*.json"))
        
        for json_file in json_files:
            base_name = json_file.stem
            pdf_file = self.input_dir / f"{base_name}.pdf"
            
            if pdf_file.exists():
                pairs.append((json_file, pdf_file))
        
        return pairs
    
    def extract_images_from_pdf(self, pdf_path: Path) -> List[bytes]:
        """Extract images from PDF file using PyMuPDF."""
        images = []
        try:
            import fitz
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
            
            doc.close()
        except Exception as e:
            self.thread_safe_print(f"Error extracting images from {pdf_path.name}: {e}")
        
        return images
    
    def describe_image_with_openai(self, image_bytes: bytes, image_index: int) -> str:
        """Generate description of image using OpenAI Vision API."""
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image or diagram in detail. If it's a flowchart or diagram, explain the flow and relationships. Be comprehensive but concise."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.thread_safe_print(f"Error describing image {image_index + 1}: {e}")
            return ""
    
    def describe_images_parallel(self, images: List[bytes]) -> List[Dict[str, Any]]:
        """Process multiple images in parallel using ThreadPoolExecutor."""
        if not images:
            return []
        
        image_descriptions = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.describe_image_with_openai, img_bytes, idx): idx
                for idx, img_bytes in enumerate(images)
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    description = future.result()
                    if description:
                        image_descriptions.append({
                            "image_index": idx,
                            "description": description
                        })
                except Exception as e:
                    self.thread_safe_print(f"Error processing image {idx + 1}: {e}")
        
        image_descriptions.sort(key=lambda x: x["image_index"])
        return image_descriptions
    
    def extract_title_from_text(self, text: str) -> str:
        """Extract title from the beginning of the extracted text."""
        if not text:
            return ""
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        title = ""
        for line in lines[:5]:
            cleaned = line.replace('#', '').replace('*', '').replace('_', '').strip()
            if '<!-- image -->' in cleaned or not cleaned:
                continue
            if len(cleaned) > 10:
                title = cleaned
                break
        
        if not title and lines:
            title = lines[0].replace('#', '').replace('*', '').strip()
        
        if len(title) > 200:
            title = title[:200] + "..."
        
        return title
    
    def process_pdf_with_docling(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF file using Docling to extract text, tables, and images."""
        result = {
            "text": "",
            "image_descriptions": [],
            "title": ""
        }
        
        try:
            conv_result = self.converter.convert(str(pdf_path))
            
            if conv_result and conv_result.document:
                text_content = conv_result.document.export_to_markdown()
                
                if text_content and text_content.strip():
                    result["text"] = text_content
                    result["title"] = self.extract_title_from_text(text_content)
        except Exception as e:
            self.thread_safe_print(f"Error processing {pdf_path.name}: {e}")
            
        
        
        if self.image_description_generation:
            images = self.extract_images_from_pdf(pdf_path)
            if images:
                result["image_descriptions"] = self.describe_images_parallel(images)
        
        return result
    
    def merge_content(self, metadata: Dict, pdf_content: Dict) -> Dict:
        """Merge PDF content with existing metadata."""
        # Create clean output with only required fields
        merged = {
            "url": metadata.get("url", ""),
            "parent_url": metadata.get("parent_url", ""),
            "title": metadata.get("title", ""),
            "description": pdf_content.get("title", metadata.get("description", "")),
            "text": "",
            "text_length": 0,
            "content_type": metadata.get("content_type", "pdf"),
            "pdf_file": metadata.get("pdf_file", "")
        }
        
        # Combine all text content
        all_text = pdf_content["text"]
        
        # Append image descriptions to text
        if pdf_content["image_descriptions"]:
            all_text += "\n\n--- Image Descriptions ---\n"
            for img_desc in pdf_content["image_descriptions"]:
                all_text += f"\nImage {img_desc['image_index'] + 1}: {img_desc['description']}\n"
        
        merged["text"] = all_text
        merged["text_length"] = len(all_text)
        
        return merged
    
    def process_all(self) -> None:
        """Process all PDF-JSON file pairs in the input directory."""
        pairs = self.find_file_pairs()
        
        if not pairs:
            log.info("No file pairs found.")
            return
        
        log.info(f"Processing {len(pairs)} file(s)...")
        
        all_documents = []
        successful = 0
        failed = 0
        
        for idx, (json_path, pdf_path) in enumerate(pairs, 1):
            log.info(f"[{idx}/{len(pairs)}] Processing {json_path.stem}...")
            try:
                if not pdf_path.exists():
                    failed += 1
                    continue
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                pdf_content = self.process_pdf_with_docling(pdf_path)
                final_data = self.merge_content(metadata, pdf_content)
                
                if final_data.get("text", "").strip():
                    all_documents.append(final_data)
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                log.error(f"Error: {e}")
        
        # Save all documents to the configured output directory
        output_file = self.output_dir / "processed_pdf.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=2, ensure_ascii=False)
        
        log.info(f"\nComplete! Success: {successful}, Failed: {failed}")
        log.info(f"Saved to: {output_file}")


if __name__ == "__main__":
    
    log.log_pipeline("Starting PDF Processing")
    
    # Load configuration
    config_loader = ConfigLoader(config_path="configs/data_processing.yaml")
    
    # Get configuration values
    INPUT_DIR = config_loader.get_unprocessed_pdf_path()
    OUTPUT_DIR = config_loader.get_processed_pdf_path()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MAX_WORKERS = 5  # You could add this to your config if needed
    IMAGE_DESCRIPTION_GENERATION = config_loader.get_description_generation_enabled()
    
    if IMAGE_DESCRIPTION_GENERATION and not OPENAI_API_KEY:
        log.error("Error: OPENAI_API_KEY not set in environment variables")
        sys.exit(1)
    
    log.log_messages([
        f"Input Directory: {INPUT_DIR}",
        f"Output Directory: {OUTPUT_DIR}",
        f"Max Workers: {MAX_WORKERS}"
    ])
    
    # Initialize preprocessor
    preprocessor = RAGPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS,
        image_description_generation=IMAGE_DESCRIPTION_GENERATION
    )
    
    # Process all PDFs
    preprocessor.process_all()
    
    log.log_pipeline("PDF Processing Completed")