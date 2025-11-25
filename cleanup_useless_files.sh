#!/bin/bash

# Cleanup script for removing useless files in the SMQF project
# Run this script to remove cache files, .DS_Store, and other temporary files

PROJECT_DIR="/Users/neilchen/Library/Mobile Documents/com~apple~CloudDocs/CodeField/Python/SAIF/SMQF"

echo "🧹 Cleaning up useless files in SMQF project..."
echo "Project directory: $PROJECT_DIR"
echo ""

# Remove __pycache__ directories
echo "Removing __pycache__ directories..."
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✓ __pycache__ directories removed"

# Remove .pyc files
echo "Removing .pyc files..."
find "$PROJECT_DIR" -type f -name "*.pyc" -delete 2>/dev/null
echo "✓ .pyc files removed"

# Remove .pyo files
echo "Removing .pyo files..."
find "$PROJECT_DIR" -type f -name "*.pyo" -delete 2>/dev/null
echo "✓ .pyo files removed"

# Remove .DS_Store files
echo "Removing .DS_Store files..."
find "$PROJECT_DIR" -type f -name ".DS_Store" -delete 2>/dev/null
echo "✓ .DS_Store files removed"

# Remove .log files
echo "Removing .log files..."
find "$PROJECT_DIR" -type f -name "*.log" -delete 2>/dev/null
echo "✓ .log files removed"

# Remove Jupyter checkpoint directories
echo "Removing Jupyter checkpoint directories..."
find "$PROJECT_DIR" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
echo "✓ Jupyter checkpoint directories removed"

# Remove pytest cache
echo "Removing pytest cache..."
find "$PROJECT_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "✓ pytest cache removed"

# Remove coverage files
echo "Removing coverage files..."
find "$PROJECT_DIR" -type f -name ".coverage" -delete 2>/dev/null
find "$PROJECT_DIR" -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null
echo "✓ coverage files removed"

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "Summary:"
echo "  - Removed Python cache files (__pycache__, *.pyc, *.pyo)"
echo "  - Removed macOS .DS_Store files"
echo "  - Removed log files"
echo "  - Removed Jupyter checkpoints"
echo "  - Removed test cache files"
