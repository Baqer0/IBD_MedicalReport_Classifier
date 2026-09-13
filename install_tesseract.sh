#!/bin/bash
set -e
if command -v tesseract >/dev/null 2>&1; then exit 0; fi
mkdir -p "$HOME/tesseract" && cd "$HOME/tesseract"
wget -q https://github.com/tesseract-ocr/tesseract/releases/download/5.3.0/tesseract-5.3.0-linux-x86_64.tar.gz
tar -xzf tesseract-5.3.0-linux-x86_64.tar.gz -C /usr/local/
test -x /usr/local/bin/tesseract
