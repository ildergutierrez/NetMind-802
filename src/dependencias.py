# =======================================================
#  librerias.py — Instalador de dependencias del proyecto
# =======================================================

import subprocess
import sys
import platform
import os

# Configurar codificación UTF-8 para evitar errores con emojis en consolas de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -------------------------------------------------------
#  LIBRERÍAS REQUERIDAS (Solo IA + Procesamiento de Archivos)
# -------------------------------------------------------
DEPENDENCIAS = [
    # --- IA / Machine Learning (Proporcionadas por la guía) ---
    "torch",
    "torchvision",
    "tokenizers",
    "pillow",
    "numpy",

    # --- PDF / Documentos (Para procesamiento de datos y entrenamiento) ---
    "pdfplumber",
    "PyMuPDF",
    "reportlab",
    "fpdf",
    "weasyprint",
    "pypdf",
    "PyPDF2",
    "python-docx",

    # --- API / Servidor Web (Para consultar la IA) ---
    "fastapi",
    "uvicorn",
    "python-multipart",

    # --- Búsqueda Web (Para enriquecer respuestas con información de Internet) ---
    "httpx",
    "huggingface_hub",

    #Directx
    "onnxruntime-directml",
]

SO = platform.system()
ES_WINDOWS = SO == "Windows"

# Definir la ruta absoluta de la carpeta 'netmind' (ubicada en la raíz del proyecto)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
VENV_DIR = os.path.join(project_root, "netmind")

# Se definirán dinámicamente en crear_venv()
VENV_PY = ""
VENV_PIP = ""

# -------------------------------------------------------
# EJECUTAR COMANDOS
# -------------------------------------------------------
def run(cmd):
    try:
        if isinstance(cmd, str):
            subprocess.check_call(cmd, shell=True)
        else:
            subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando: {cmd}\n{e}")

# -------------------------------------------------------
# CREAR VENV
# -------------------------------------------------------
def crear_venv():
    global VENV_PY
    global VENV_PIP

    if ES_WINDOWS:
        VENV_PY = os.path.join(VENV_DIR, "Scripts", "python.exe")
        VENV_PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        VENV_PY = os.path.join(VENV_DIR, "bin", "python")
        VENV_PIP = os.path.join(VENV_DIR, "bin", "pip")

    if not os.path.exists(VENV_DIR):
        print("🟦 Creando entorno virtual...")
        
        # Verificar si venv está disponible en Linux (especialmente Debian/Ubuntu)
        try:
            import venv
            venv_disponible = True
        except ImportError:
            venv_disponible = False

        if not venv_disponible:
            if SO == "Linux":
                print("⚠️  Módulo 'venv' no disponible. Intentando instalar 'python3-venv'...")
                run("sudo apt update && sudo apt install -y python3-venv")
            else:
                print("❌ El módulo 'venv' de Python no está disponible.")
                sys.exit(1)

        run([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("🟩 Entorno virtual ya existe")

# -------------------------------------------------------
# VERIFICAR SI EL PAQUETE ESTÁ INSTALADO
# -------------------------------------------------------
def paquete_instalado(python_path, paquete):
    try:
        subprocess.check_output(
            [python_path, "-m", "pip", "show", paquete],
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

# -------------------------------------------------------
# INSTALAR DEPENDENCIAS
# -------------------------------------------------------
def instalar_dependencias(python_path, pip_path):
    print("\n📦 Verificando dependencias...\n")
    
    # Actualizar herramientas base
    print("🔄 Actualizando pip, setuptools y wheel...")
    run([python_path, "-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel"])

    for pkg in DEPENDENCIAS:
        if paquete_instalado(python_path, pkg):
            print(f"  ✅ {pkg} ya está instalado")
        else:
            print(f"  📥 Instalando {pkg}...")
            run([python_path, "-m", "pip", "install", pkg])
            
    print("\n✅ Todas las dependencias requeridas están verificadas e instaladas")

# -------------------------------------------------------
# FLUJO PRINCIPAL
# -------------------------------------------------------
def configurar_entorno():
    print("🚀 Iniciando configuración de dependencias...\n")

    # CREAR VENV
    crear_venv()

    python = VENV_PY
    pip = VENV_PIP

    # VERIFICAR VENV
    if not os.path.exists(python):
        print("❌ No se encontró el entorno virtual")
        return

    # INSTALAR DEPENDENCIAS
    instalar_dependencias(python, pip)

    # MOSTRAR PYTHON
    print("\n🐍 Python en uso:")
    run([python, "--version"])

    print(f"\n🎉 Entorno configurado correctamente en {SO}")

if __name__ == "__main__":
    configurar_entorno()
