#!/bin/bash

# BEET Bot Runner Script
# Usage: ./run_beet_bot.sh
# Simple launcher for BEET Virtual Microphone Application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$SCRIPT_DIR/venv"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[BEET BOT]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[BEET BOT]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[BEET BOT]${NC} $1"
}

print_error() {
    echo -e "${RED}[BEET BOT]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "BEET Bot - Virtual Microphone Application"
    echo ""
    echo "Usage: $0"
    echo ""
    echo "Features:"
    echo "  • Virtual Microphone creation and control"
    echo "  • Text-to-Speech (TTS) with offline and online options"
    echo "  • Speech-to-Text (STT) using Groq API"
    echo "  • Meeting join functionality (Google Meet, Teams, Zoom)"
    echo "  • Audio device management"
    echo ""
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check if PulseAudio is available
    if ! command -v pactl &> /dev/null; then
        print_error "PulseAudio (pactl) is required but not found"
        print_error "Install with: sudo apt-get install pulseaudio"
        exit 1
    fi
    
    # Check if PulseAudio is running
    if ! pulseaudio --check; then
        print_warning "PulseAudio is not running, attempting to start..."
        pulseaudio --start --daemon || {
            print_error "Failed to start PulseAudio"
            exit 1
        }
    fi
    
    print_success "Dependencies check passed"
}

# Function to setup virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements if they exist
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_status "Installing Python dependencies..."
        pip install -r "$REQUIREMENTS_FILE"
    else
        print_warning "No requirements.txt found, installing basic dependencies..."
        pip install PyQt5 playwright pulsectl numpy pyttsx3 gtts pygame groq sounddevice scipy python-dotenv
    fi
    
    # Install Playwright browsers
    print_status "Installing Playwright browsers..."
    playwright install chromium || print_warning "Playwright browser installation failed"
    
    print_success "Virtual environment setup complete"
}

# Function to launch main GUI
launch_gui() {
    print_status "Launching BEET Bot GUI..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Set environment variables
    export DISPLAY="${DISPLAY:-:0}"
    
    # Launch the main Python application
    if [ -f "$MAIN_PY" ]; then
        python3 "$MAIN_PY"
    else
        print_error "main.py not found in $SCRIPT_DIR"
        exit 1
    fi
}

# Main execution
main() {
    # Parse command line arguments
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    fi
    
    print_success "🚀 BEET Bot Starting..."
    echo ""
    
    # Check system dependencies
    check_dependencies
    
    # Setup Python environment
    setup_venv
    
    echo ""
    print_success "🎯 Launching BEET Bot GUI!"
    echo ""
    
    # Launch the GUI
    launch_gui
}

# Run main function
main "$@" 