import yaml
from typing import List, Dict, Any

@staticmethod
def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return {}

@staticmethod   
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