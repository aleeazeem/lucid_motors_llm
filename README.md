# This files have two pipelines
# 1. RAG Web Crawler & Document Processing Pipeline

A complete RAG (Retrieval-Augmented Generation) pipeline that crawls websites, processes PDFs with image descriptions, and creates embeddings stored in Milvus vector database.

## Features

- Web crawling with configurable depth
- PDF extraction and processing using Docling
- Optional AI-powered image description generation (GPT-4o)
- Embeddings generation and storage in Milvus
- Concurrent processing for performance
- Docker Compose and standalone Milvus support

## Project Structure

```
.
├── configs/
│   └── data_processing.yaml    # Main configuration file
├── src/
│   ├── core/
│   │   ├── base.py                    # ConfigLoader class
│   │   ├── web_crawler_process.py     # Web crawler
│   │   ├── web_pdf_preprocessor.py    # PDF processor
│   │   └── create_embeddings.py       # Embeddings generator
│   └── utils/
│       ├── logger.py                  # Logging utilities
│       ├── loader_utils.py            # YAML loader
│       └── web_crawler_utils.py       # Crawler utilities
├── output/                     # Generated output directory
├── docker-compose.yml          # Milvus Docker setup
├── setup.sh                    # Main setup and run script
├── stop.sh                     # Stop Milvus script
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables
```

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (recommended). If you run setup.bash file then it will be taken care
- OpenAI API key (optional, only if `description_generation: True`)


### Configure Your Project
The file contains all the configuration of the project `configs/data_processing.yaml`:


### Quick Start (Automated)

Run the complete pipeline with one command:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
1. Install Python dependencies
2. Check and start Milvus (Docker Compose or standalone)
3. Crawl web pages and download PDFs
4. Process PDFs and extract text/images
5. Generate embeddings and store in Milvus
6. Run Evaluation

