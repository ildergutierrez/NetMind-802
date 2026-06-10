"""
=======================================================
 api.py — Punto de entrada principal de NetMind-802
=======================================================
 Flujo al ejecutar:
   1. Verifica e instala dependencias (dependencias.py)
   2. Levanta el servidor FastAPI en http://127.0.0.1:8000
   3. Verifica si existe el modelo entrenado. Si no, entrena.
   4. Responde consultas de diagnóstico de red en tiempo real.
   5. Aprende de los casos enviados (POST /feedback).
   6. Puede enriquecer respuestas con búsqueda web (DuckDuckGo).
=======================================================
"""

import os
import sys
import json
import shutil
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager

# -------------------------------------------------------
# CODIFICACIÓN UTF-8 EN WINDOWS
# -------------------------------------------------------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -------------------------------------------------------
# PASO 1: INSTALAR / VERIFICAR DEPENDENCIAS
# -------------------------------------------------------
from dependencias import configurar_entorno
configurar_entorno()

# -------------------------------------------------------
# PASO 2: IMPORTAR LIBRERÍAS DEL SERVIDOR
# -------------------------------------------------------
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------
# PASO 3: IMPORTAR MÓDULOS INTERNOS DEL PROYECTO
# -------------------------------------------------------
from test_inference import diagnosticar
from train import entrenar_modelo
from web_search import construir_contexto_web

# -------------------------------------------------------
# RUTAS PRINCIPALES DEL PROYECTO
# -------------------------------------------------------
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
dataset_path      = os.path.join(project_root, "data", "dataset_redes.json")
corpus_path       = os.path.join(project_root, "data", "corpus_redes.txt")
checkpoint_path   = os.path.join(project_root, "models", "netmind_vlm.pth")
tokenizer_path    = os.path.join(current_dir, "tokenizer_redes.json")

# -------------------------------------------------------
# DETECTAR DISPOSITIVO DE CÓMPUTO (GPU / CPU)
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# EVENTO DE INICIO: Verificar modelo al levantar el server
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*52)
    print("   NetMind-802 — IA de Redes IEEE 802")
    print("="*52)
    print(f"   Dispositivo de inferencia : {device.upper()}")
    print(f"   Dataset                   : {dataset_path}")
    print(f"   Tokenizador               : {tokenizer_path}")
    print(f"   Modelo                    : {checkpoint_path}")
    print("="*52)

    if os.path.exists(checkpoint_path):
        print("   Modelo entrenado encontrado. Listo.")
    else:
        print("   Modelo no encontrado. Iniciando entrenamiento...")
        entrenar_modelo()
        print("   Entrenamiento completado. Modelo guardado.")

    print("="*52 + "\n")
    yield
    print("\n   Servidor NetMind-802 detenido.")

# -------------------------------------------------------
# INSTANCIA PRINCIPAL DE FASTAPI
# -------------------------------------------------------
app = FastAPI(
    title="NetMind-802 API",
    description=(
        "API REST de la IA especializada en redes cableadas e inalámbricas (IEEE 802), "
        "infraestructura de red, telemática y seguridad multiplataforma. "
        "Compatible con Cisco IOS, Windows PowerShell y Linux Bash."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------------------------------------
# CORS — Acepta consultas desde cualquier origen
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
#  ENDPOINTS
# =======================================================

# -------------------------------------------------------
# GET /
# Bienvenida y mapa de endpoints
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "proyecto"   : "NetMind-802",
        "descripcion": "IA especializada en redes IEEE 802, telemática y seguridad.",
        "version"    : "1.0.0",
        "endpoints"  : {
            "GET  /status"      : "Estado del servidor y del modelo",
            "POST /diagnose"    : "Diagnosticar un problema de red (texto + imagen opcional)",
            "POST /diagnose-web": "Diagnosticar + enriquecer con búsqueda web (DuckDuckGo)",
            "POST /feedback"    : "Enviar un caso nuevo para que la IA aprenda de él",
            "POST /train"       : "Re-entrenar el modelo con el dataset actualizado",
        }
    }

# -------------------------------------------------------
# GET /status
# Estado del servidor y disponibilidad del modelo
# -------------------------------------------------------
@app.get("/status")
def get_status():
    modelo_listo = os.path.exists(checkpoint_path)

    # Contar casos en el dataset
    total_casos = 0
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_casos = len(data)
        except Exception:
            pass

    return {
        "status"           : "online",
        "device"           : device.upper(),
        "modelo_entrenado" : modelo_listo,
        "casos_en_dataset" : total_casos,
        "mensaje"          : (
            f"Servidor activo. Modelo listo con {total_casos} caso(s) de entrenamiento."
            if modelo_listo else
            "Servidor activo. Usa POST /train para entrenar el modelo."
        )
    }

# -------------------------------------------------------
# POST /diagnose
# Diagnóstico de red con solo el modelo local
# -------------------------------------------------------
@app.post("/diagnose")
async def api_diagnose(
    prompt: str = Form(..., description="Descripción del problema de red, logs o comandos a analizar."),
    image : UploadFile = File(None, description="Imagen opcional: topología, captura de Wireshark, etc.")
):
    """
    Diagnóstico local usando únicamente el modelo entrenado.
    Retorna: Diagnóstico, Causas y Comandos de Solución.
    """
    if not os.path.exists(checkpoint_path):
        raise HTTPException(
            status_code=503,
            detail="El modelo no está entrenado. Llama a POST /train primero."
        )

    temp_img_path = None
    if image and image.filename:
        ext = os.path.splitext(image.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado.")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(image.file, tmp)
                temp_img_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {e}")

    try:
        respuesta = diagnosticar(prompt, image_path=temp_img_path, device=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante inferencia: {e}")
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass

    return {
        "status"       : "success",
        "device"       : device.upper(),
        "tiene_imagen" : temp_img_path is not None,
        "fuente"       : "modelo_local",
        "prompt"       : prompt,
        "respuesta"    : respuesta
    }

# -------------------------------------------------------
# POST /diagnose-web
# Diagnóstico enriquecido con búsqueda en Internet
# -------------------------------------------------------
@app.post("/diagnose-web")
async def api_diagnose_web(
    prompt: str = Form(..., description="Descripción del problema de red a diagnosticar y buscar en la web."),
    image : UploadFile = File(None, description="Imagen opcional: topología o captura.")
):
    """
    Diagnóstico híbrido: modelo local + búsqueda web (DuckDuckGo).
    Primero consulta el modelo entrenado y luego enriquece la respuesta
    con información actualizada de internet.
    """
    if not os.path.exists(checkpoint_path):
        raise HTTPException(
            status_code=503,
            detail="El modelo no está entrenado. Llama a POST /train primero."
        )

    temp_img_path = None
    if image and image.filename:
        ext = os.path.splitext(image.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado.")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(image.file, tmp)
                temp_img_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {e}")

    try:
        # Respuesta del modelo local
        respuesta_modelo = diagnosticar(prompt, image_path=temp_img_path, device=device)

        # Enriquecer con búsqueda web
        contexto_web = construir_contexto_web(prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass

    return {
        "status"          : "success",
        "device"          : device.upper(),
        "tiene_imagen"    : temp_img_path is not None,
        "fuente"          : "modelo_local + web",
        "prompt"          : prompt,
        "respuesta_modelo": respuesta_modelo,
        "contexto_web"    : contexto_web if contexto_web else "Sin resultados adicionales de la web."
    }

# -------------------------------------------------------
# POST /feedback
# El usuario envía un caso correcto → la IA lo aprende
# -------------------------------------------------------
@app.post("/feedback")
async def api_feedback(
    prompt  : str = Form(..., description="El prompt o problema de red que se describió."),
    respuesta: str = Form(..., description="La respuesta correcta (Diagnóstico + Causas + Solución)."),
    titulo  : str = Form("Sin título", description="Título opcional del caso para identificarlo.")
):
    """
    Permite al usuario enviar un caso correcto para que la IA lo aprenda.
    El caso se guarda en data/dataset_redes.json y puede disparar
    un re-entrenamiento automático con POST /train.
    """
    if not prompt.strip() or not respuesta.strip():
        raise HTTPException(
            status_code=400,
            detail="El prompt y la respuesta son obligatorios y no pueden estar vacíos."
        )

    # Cargar dataset actual
    try:
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        else:
            dataset = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer dataset: {e}")

    # Generar un ID único basado en la fecha y hora actual
    nuevo_id = f"caso_fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    nuevo_caso = {
        "id"       : nuevo_id,
        "titulo"   : titulo.strip(),
        "prompt"   : prompt.strip(),
        "target"   : respuesta.strip(),
        "origen"   : "feedback_usuario",
        "fecha"    : datetime.now().isoformat()
    }

    dataset.append(nuevo_caso)

    # Guardar dataset actualizado
    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar en dataset: {e}")

    # También agregar el prompt al corpus para actualizar el vocabulario
    try:
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n{prompt}\n{respuesta}\n")
    except Exception:
        pass  # No es crítico si falla

    return {
        "status"      : "success",
        "mensaje"     : f"Caso '{nuevo_id}' guardado. El modelo aprenderá en el próximo entrenamiento.",
        "id_caso"     : nuevo_id,
        "total_casos" : len(dataset),
        "sugerencia"  : "Llama a POST /train para re-entrenar el modelo con el nuevo caso."
    }

# -------------------------------------------------------
# POST /train
# Re-entrenar el modelo con el dataset actualizado
# -------------------------------------------------------
@app.post("/train")
def api_train():
    """
    Re-entrena el modelo con el dataset actual (incluyendo
    los casos enviados por /feedback). Tarda algunos minutos en CPU.
    """
    try:
        entrenar_modelo()
        return {
            "status"     : "success",
            "mensaje"    : "Modelo re-entrenado y guardado exitosamente.",
            "modelo_path": checkpoint_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el entrenamiento: {e}")


# =======================================================
# PUNTO DE ENTRADA
# =======================================================
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
