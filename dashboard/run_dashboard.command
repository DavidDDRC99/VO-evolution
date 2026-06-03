#!/bin/bash
cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creant entorn virtual..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Instal·lant dependències..."
pip install -r requirements.txt --quiet

echo ""
echo "=== VO-evolution Dashboard ==="
echo "Obre http://127.0.0.1:8050 al teu navegador"
echo "Prem Ctrl+C per aturar"
echo ""
python app.py

echo ""
echo "Dashboard aturat."
read -p "Prem Enter per tancar..."
