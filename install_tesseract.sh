#!/bin/bash
set -e

TESSERACT_DIR="/usr/local/bin"

if command -v tesseract >/dev/null 2>&1; then
    echo "Tesseract already installed:"
    tesseract --version
    exit 0
fi

apt-get update
apt-get install -y tesseract-ocr

echo "Tesseract installed:"
tesseract --version
