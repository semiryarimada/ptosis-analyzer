#!/bin/bash
cd "$(dirname "$0")"

# Check Python3
if ! command -v python3 &>/dev/null; then
    osascript -e 'display alert "Python 3 bulunamadı" message "https://www.python.org/downloads/ adresinden Python 3.10+ yükleyin." buttons {"Tamam"} default button 1'
    open "https://www.python.org/downloads/"
    exit 1
fi

python3 launcher.py
