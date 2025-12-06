#!/bin/bash
# Setup script for research assistant

set -e

echo "=================================="
echo "Research Assistant Setup"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Checking prerequisites..."
echo "=================================="

# Check Tesseract
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract OCR is installed: $(tesseract --version | head -n 1)"
else
    echo "✗ Tesseract OCR not found"
    echo "  Install with: brew install tesseract"
    echo ""
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed: $(docker --version)"
else
    echo "✗ Docker not found"
    echo "  Install from: https://www.docker.com/products/docker-desktop"
    echo ""
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed: $(ollama --version 2>&1 || echo 'version unknown')"
else
    echo "✗ Ollama not found"
    echo "  Install from: https://ollama.ai"
    echo ""
fi

echo ""
echo "=================================="
echo "Pulling required Ollama models..."
echo "=================================="

if command -v ollama &> /dev/null; then
    echo "Pulling deepseek-r1:8b..."
    ollama pull deepseek-r1:8b || echo "Failed to pull deepseek-r1:8b"

    echo ""
    echo "Pulling nomic-embed-text..."
    ollama pull nomic-embed-text || echo "Failed to pull nomic-embed-text"
else
    echo "Skipping (Ollama not installed)"
fi

echo ""
echo "=================================="
echo "Starting GROBID service..."
echo "=================================="

if command -v docker &> /dev/null; then
    # Check if GROBID container exists
    if docker ps -a --format '{{.Names}}' | grep -q '^grobid$'; then
        # If exists, ensure it's running
        if docker ps --format '{{.Names}}' | grep -q '^grobid$'; then
            echo "GROBID is already running"
        else
            echo "Starting existing GROBID container..."
            docker start grobid
            echo "GROBID started on port 8070"
        fi
    else
        echo "Starting new GROBID Docker container..."
        docker run -d -p 8070:8070 --name grobid lfoppiano/grobid:0.8.0
        echo "GROBID started on port 8070"
    fi
else
    echo "Skipping (Docker not installed)"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the pipeline:"
echo "     python cli.py process --root-dir /path/to/papers --topic 'Your research topic'"
echo ""
echo "  3. View help:"
echo "     python cli.py process --help"
echo ""
