#!/bin/bash
# Interactive setup script for Research Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║           Research Assistant Setup Wizard                 ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=macOS;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM=Windows;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac

echo -e "${BLUE}Detected platform: ${PLATFORM}${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ========================================
# Step 1: Check Python
# ========================================
echo -e "${YELLOW}[Step 1/5] Checking Python installation...${NC}"
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
    
    # Check Python version >= 3.12
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 12 ]); then
        echo -e "${RED}✗ Python 3.12+ required, found ${PYTHON_VERSION}${NC}"
        echo "  Please upgrade Python: https://www.python.org/downloads/"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.12+: https://www.python.org/downloads/"
    exit 1
fi
echo ""

# ========================================
# Step 2: Check Tesseract (OCR)
# ========================================
echo -e "${YELLOW}[Step 2/5] Checking Tesseract OCR...${NC}"
if command_exists tesseract; then
    TESSERACT_VERSION=$(tesseract --version 2>&1 | head -n 1)
    echo -e "${GREEN}✓ ${TESSERACT_VERSION}${NC}"
else
    echo -e "${RED}✗ Tesseract not found${NC}"
    echo "  Tesseract is required for OCR (extracting text from scanned PDFs)"
    echo ""
    if [ "$PLATFORM" = "macOS" ]; then
        echo "  Install with: brew install tesseract"
    elif [ "$PLATFORM" = "Linux" ]; then
        echo "  Install with: sudo apt-get install tesseract-ocr"
    else
        echo "  Install from: https://github.com/tesseract-ocr/tesseract"
    fi
    echo ""
    read -p "  Continue without Tesseract? (OCR will not work) [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# ========================================
# Step 3: Choose LLM Provider
# ========================================
echo -e "${YELLOW}[Step 3/5] Configure LLM Provider${NC}"
echo ""
echo "Choose your LLM provider:"
echo "  1) Ollama (Local, Free, Private - Recommended)"
echo "  2) Google Gemini (Cloud, Requires API Key)"
echo "  3) Both (I'll decide at runtime)"
echo ""
read -p "Enter choice [1/2/3]: " -n 1 -r LLM_CHOICE
echo ""
echo ""

USE_OLLAMA=false
USE_GEMINI=false
GEMINI_KEY=""

case $LLM_CHOICE in
    1)
        USE_OLLAMA=true
        ;;
    2)
        USE_GEMINI=true
        ;;
    3)
        USE_OLLAMA=true
        USE_GEMINI=true
        ;;
    *)
        echo -e "${RED}Invalid choice. Defaulting to Ollama.${NC}"
        USE_OLLAMA=true
        ;;
esac

# ========================================
# Step 3a: Setup Ollama
# ========================================
if [ "$USE_OLLAMA" = true ]; then
    echo -e "${YELLOW}[3a] Setting up Ollama...${NC}"
    
    if command_exists ollama; then
        echo -e "${GREEN}✓ Ollama is installed${NC}"
        
        # Check if ollama is running
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama server is running${NC}"
        else
            echo -e "${YELLOW}⚠ Ollama is not running${NC}"
            echo "  Starting Ollama server..."
            if [ "$PLATFORM" = "macOS" ]; then
                open -a Ollama 2>/dev/null || echo "  Please start Ollama manually"
            else
                echo "  Please start Ollama: ollama serve"
            fi
            echo "  Waiting for server to start..."
            sleep 3
        fi
        
        # Check required models
        echo ""
        echo "Checking required models..."
        MODELS=$(ollama list 2>/dev/null || echo "")
        
        if echo "$MODELS" | grep -q "deepseek-r1:8b"; then
            echo -e "${GREEN}✓ deepseek-r1:8b found${NC}"
        else
            echo -e "${YELLOW}⚠ deepseek-r1:8b not found${NC}"
            read -p "  Download deepseek-r1:8b now? (~4.7GB) [Y/n]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                ollama pull deepseek-r1:8b
            fi
        fi
        
        if echo "$MODELS" | grep -q "nomic-embed-text"; then
            echo -e "${GREEN}✓ nomic-embed-text found${NC}"
        else
            echo -e "${YELLOW}⚠ nomic-embed-text not found${NC}"
            read -p "  Download nomic-embed-text now? (~274MB) [Y/n]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                ollama pull nomic-embed-text
            fi
        fi
        
    else
        echo -e "${RED}✗ Ollama not installed${NC}"
        echo ""
        echo "  Ollama provides free, local LLMs (no API keys needed)"
        echo "  Download from: https://ollama.com/download"
        echo ""
        if [ "$PLATFORM" = "macOS" ]; then
            echo "  macOS: Download the .dmg and install"
        elif [ "$PLATFORM" = "Linux" ]; then
            echo "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
        fi
        echo ""
        read -p "  Continue without Ollama? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""
fi

# ========================================
# Step 3b: Setup Gemini API Key
# ========================================
if [ "$USE_GEMINI" = true ]; then
    echo -e "${YELLOW}[3b] Setting up Google Gemini API...${NC}"
    echo ""
    
    # Check if key already exists
    if [ -n "$GEMINI_API_KEY" ]; then
        echo -e "${GREEN}✓ GEMINI_API_KEY already set in environment${NC}"
        echo "  Current key: ${GEMINI_API_KEY:0:10}...${GEMINI_API_KEY: -4}"
        read -p "  Use this key? [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            GEMINI_KEY="$GEMINI_API_KEY"
        fi
    fi
    
    if [ -z "$GEMINI_KEY" ]; then
        echo "Get your free API key from: ${BLUE}https://aistudio.google.com/app/apikey${NC}"
        echo ""
        read -p "Paste your Gemini API key (or press Enter to skip): " GEMINI_KEY
        echo ""
        
        if [ -n "$GEMINI_KEY" ]; then
            echo -e "${GREEN}✓ Gemini API key received${NC}"
        else
            echo -e "${YELLOW}⚠ No API key provided - you'll need to set it later${NC}"
        fi
    fi
    echo ""
fi

# ========================================
# Step 4: Create .env file
# ========================================
echo -e "${YELLOW}[Step 4/5] Creating configuration files...${NC}"

# Create .env file
if [ -n "$GEMINI_KEY" ]; then
    echo "GEMINI_API_KEY=$GEMINI_KEY" > .env
    echo -e "${GREEN}✓ Created .env file with Gemini API key${NC}"
    
    # Add to shell profile for persistence
    SHELL_PROFILE=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    fi
    
    if [ -n "$SHELL_PROFILE" ]; then
        read -p "Add GEMINI_API_KEY to $SHELL_PROFILE permanently? [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            if ! grep -q "GEMINI_API_KEY" "$SHELL_PROFILE"; then
                echo "" >> "$SHELL_PROFILE"
                echo "# Research Assistant - Gemini API Key" >> "$SHELL_PROFILE"
                echo "export GEMINI_API_KEY=$GEMINI_KEY" >> "$SHELL_PROFILE"
                echo -e "${GREEN}✓ Added to $SHELL_PROFILE${NC}"
                echo "  Run: source $SHELL_PROFILE"
            else
                echo -e "${YELLOW}⚠ GEMINI_API_KEY already in $SHELL_PROFILE${NC}"
            fi
        fi
    fi
fi

# Create example config.yaml
if [ ! -f "config.yaml" ]; then
    cat > config.yaml << EOF
# Research Assistant Configuration

# LLM Provider: ollama or gemini
llm_provider: ollama

# Scoring thresholds
scoring:
  min_topic_relevance: 5  # Papers below this go to quarantined/ (1-10 scale)

# Deduplication
dedup:
  similarity_threshold: 0.95
  use_minhash: true
  num_perm: 128

# Ollama configuration
ollama:
  summarize_model: "deepseek-r1:8b"
  classify_model: "deepseek-r1:8b"
  embed_model: "nomic-embed-text"
  temperature: 0.1
  base_url: "http://localhost:11434"

# Gemini configuration
gemini:
  api_key: "\${GEMINI_API_KEY}"  # References environment variable
  temperature: 0.1

# Rate limiting (Gemini API)
rate_limit:
  enabled: true
  rpm_limit: 10   # Requests per minute
  rpd_limit: 500  # Requests per day

# Processing
processing:
  workers: 2  # Parallel workers
  batch_size: 32
EOF
    echo -e "${GREEN}✓ Created config.yaml${NC}"
fi

echo ""

# ========================================
# Step 5: Installation Examples
# ========================================
echo -e "${YELLOW}[Step 5/5] Setup Complete!${NC}"
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗"
echo "║                   Setup Successful!                       ║"
echo "╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Quick Start Examples:${NC}"
echo ""

if [ "$USE_OLLAMA" = true ]; then
    echo -e "${YELLOW}# Using Ollama (local, free):${NC}"
    echo "research-assistant process \\"
    echo "  --root-dir ~/Documents/papers \\"
    echo "  --topic \"Prompt Injection Attacks in LLMs\" \\"
    echo "  --llm-provider ollama \\"
    echo "  --workers 2"
    echo ""
fi

if [ "$USE_GEMINI" = true ] && [ -n "$GEMINI_KEY" ]; then
    echo -e "${YELLOW}# Using Gemini (cloud, fast):${NC}"
    echo "research-assistant process \\"
    echo "  --root-dir ~/Documents/papers \\"
    echo "  --topic \"Prompt Injection Attacks in LLMs\" \\"
    echo "  --llm-provider gemini \\"
    echo "  --workers 2"
    echo ""
fi

echo -e "${YELLOW}# Dry run (no file moves):${NC}"
echo "research-assistant process \\"
echo "  --root-dir ~/Documents/papers \\"
echo "  --topic \"Your Research Topic\" \\"
echo "  --dry-run"
echo ""

echo -e "${YELLOW}# Resume interrupted run:${NC}"
echo "research-assistant process \\"
echo "  --root-dir ~/Documents/papers \\"
echo "  --topic \"Your Research Topic\" \\"
echo "  --resume"
echo ""

echo -e "${YELLOW}# Custom output location:${NC}"
echo "research-assistant process \\"
echo "  --root-dir ~/Documents/papers \\"
echo "  --topic \"Your Research Topic\" \\"
echo "  --output-dir ~/Desktop"
echo ""

echo -e "${BLUE}More Help:${NC}"
echo "  research-assistant --help"
echo "  research-assistant process --help"
echo ""

echo -e "${BLUE}Documentation:${NC}"
echo "  README: https://github.com/rexmirak/research_assistant"
echo ""

echo -e "${GREEN}Happy researching! 🎓📚${NC}"
