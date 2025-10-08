import yaml
import json
from typing import List, Dict, Any
from src.utils.logger import Logger
log = Logger()

@staticmethod
def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        log.info(f"Error: File '{file_path}' not found")
        return {}
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML: {e}")
        return {}

@staticmethod   
def load_json(json_file_path: str) -> list:
    """
    Load JSON file and return documents
    
    Args:
        json_file_path (str): Path to JSON file
        
    Returns:
        list: List of documents from the JSON file, empty list if error
    """
    try:
        log.info(f"Loading: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
            log.info(f"  Loaded {len(docs)} documents")
            return docs
    except FileNotFoundError:
        log.error(f"  ERROR: File not found - {json_file_path}")
        return []
    except json.JSONDecodeError:
        log.error(f"  ERROR: Invalid JSON - {json_file_path}")
        return []
    except Exception as e:
        log.error(f"  ERROR: {e}")
        return []
    
@staticmethod
def load_all_json_files(json_file_paths: list) -> list:
    """
    Load multiple JSON files and combine all documents
    
    Args:
        json_file_paths (list): List of paths to JSON files
        
    Returns:
        list: Combined list of all documents from all files
    """
    all_documents = []
    for json_file in json_file_paths:
        docs = load_json(json_file)
        all_documents.extend(docs)
    if not all_documents:
        log.info("\nNo documents loaded from any files.")
    else:
        log.info(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


@staticmethod
def load_all_files(folder_path: str, extension: str) -> list:
    """
    Load all JSON files from a folder and combine all documents
    
    Args:
        folder_path (str): Path to folder containing JSON files
        
    Returns:
        list: Combined list of all documents from all files
    """
    import os
    from pathlib import Path
    
    all_documents = []
    folder = Path(folder_path)
    
    # Get all JSON files in the folder
    pattern = f"*.{extension.lstrip('.')}"
    json_files = list(folder.glob(pattern))
    
    if not json_files:
        log.info(f"No JSON files found in: {folder_path}")
        return all_documents
    log.info(f"Found {len(json_files)} JSON file(s) in {folder_path}")
    
    # Load each JSON file
    for json_file in json_files:
        try:
            docs = load_json(str(json_file))
            all_documents.extend(docs)
            log.info(f"  Loaded {len(docs)} documents from {json_file.name}")
        except Exception as e:
            log.error(f"  Error loading {json_file.name}: {e}")
    
    if not all_documents:
        log.info("No documents loaded from any files.")
    else:
        log.info(f"Total documents loaded: {len(all_documents)}")
    
    return all_documents