"""
=======================================================
 main.py — Punto de entrada del proyecto NetMind-802
 Redirige la ejecución al servidor principal: api.py
=======================================================
"""
import os
import sys

# Asegurar que el directorio src esté en el path de Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Invocar directamente el servidor principal
from api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
