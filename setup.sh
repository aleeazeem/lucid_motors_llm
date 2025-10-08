#!/bin/bash

set -e  # Exit on error

################################# Install Dependencies ########################################
echo "Setting up RAG project on MacBook..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
################################# Install Dependencies ########################################



################################## Milvus setup ###############################################
# Check if Milvus is already running 
check_milvus() {
    if curl -s http://localhost:19530 > /dev/null 2>&1; then
        return 0  # Milvus is running
    else
        return 1  # Milvus is not running
    fi
}

# Check if Docker is available
check_docker() {
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Start Milvus if not running
if check_milvus; then
    echo "✓ Milvus is already running at localhost:19530"
else
    echo "Milvus not detected. Starting Milvus..."
    
    # Prefer Docker Compose if available and docker-compose.yml exists
    if check_docker && [ -f "docker-compose.yml" ]; then
        echo "Using Docker Compose to start Milvus..."
        
        # Check if containers are already created but stopped
        if docker ps -a | grep -q "milvus-standalone"; then
            echo "Starting existing Milvus containers..."
            docker-compose start
        else
            echo "Creating and starting Milvus containers..."
            docker-compose up -d
        fi
        
        # Wait for Milvus to be ready (with timeout)
        echo "Waiting for Milvus to start..."
        TIMEOUT=90
        ELAPSED=0
        while ! check_milvus; do
            if [ $ELAPSED -ge $TIMEOUT ]; then
                echo "Error: Milvus failed to start within ${TIMEOUT} seconds"
                echo "Check logs with: docker-compose logs"
                exit 1
            fi
            sleep 3
            ELAPSED=$((ELAPSED + 3))
            echo "  Waiting... (${ELAPSED}s)"
        done
        
        echo "✓ Milvus started successfully via Docker Compose at localhost:19530"
        
    else
        # Fallback to standalone script
        if ! check_docker; then
            echo "Docker not found. Using standalone Milvus..."
        else
            echo "docker-compose.yml not found. Using standalone Milvus..."
        fi
        
        # Check if standalone_embed.sh exists
        if [ ! -f "standalone_embed.sh" ]; then
            echo "Downloading Milvus standalone script..."
            curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
            chmod +x standalone_embed.sh
        fi
        
        # Start Milvus
        bash standalone_embed.sh start
        
        # Wait for Milvus to be ready (with timeout)
        echo "Waiting for Milvus to start..."
        TIMEOUT=60
        ELAPSED=0
        while ! check_milvus; do
            if [ $ELAPSED -ge $TIMEOUT ]; then
                echo "Error: Milvus failed to start within ${TIMEOUT} seconds"
                exit 1
            fi
            sleep 2
            ELAPSED=$((ELAPSED + 2))
            echo "  Waiting... (${ELAPSED}s)"
        done
        
        echo "✓ Milvus started successfully via standalone at localhost:19530"
    fi
fi
################################## Milvus setup ###############################################


################################## Imgestion Pipeline #########################################
# Crawl through HTML pages
echo "Step 1: Crawling web pages..."
python3 src/core/web_crawler_process.py

# Process PDF files
echo ""
echo "Step 2: Processing PDF files..."
python3 src/core/web_pdf_process.py

# Create embeddings and save to Milvus
echo ""
echo "Step 3: Creating embeddings and storing in Milvus..."
python3 src/core/ingestion.py

# Run evaluation
echo ""
echo "Step 4: Running evaluation..."
python3 src/core/evaluation.py

echo ""
echo "✓ RAG pipeline completed successfully!"

################################## Imgestion Pipeline #########################################