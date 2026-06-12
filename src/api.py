"""
=======================================================
 api.py — Punto de entrada principal de NetMind-802
=======================================================
 Flujo al ejecutar:
   1. Verifica e instala dependencias (dependencias.py)
   2. Levanta el servidor FastAPI en http://127.0.0.1:8000
   3. Verifica si existe el modelo entrenado. Si no, entrena.
   4. Carga el modelo UNA SOLA VEZ en memoria al iniciar.
   5. Responde consultas de diagnóstico de red en tiempo real.
   6. Aprende de los casos enviados (POST /feedback).
   7. Puede enriquecer respuestas con búsqueda web (DuckDuckGo).
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
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------
# PASO 3: IMPORTAR MÓDULOS INTERNOS DEL PROYECTO
# -------------------------------------------------------
from test_inference import cargar_modelo, diagnosticar
from train import entrenar_modelo
from web_search import construir_contexto_web

# -------------------------------------------------------
# RUTAS PRINCIPALES DEL PROYECTO
# -------------------------------------------------------
current_dir     = os.path.dirname(os.path.abspath(__file__))
project_root    = os.path.dirname(current_dir)
dataset_path    = os.path.join(project_root, "data",   "dataset_redes.json")
corpus_path     = os.path.join(project_root, "data",   "corpus_redes.txt")
checkpoint_path = os.path.join(project_root, "models", "netmind_vlm.pth")
tokenizer_path  = os.path.join(current_dir,            "tokenizer_redes.json")

# -------------------------------------------------------
# EVENTO DE INICIO: Entrenar si hace falta y cargar modelo
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*55)
    print("   NetMind-802 — IA de Redes IEEE 802")
    print("="*55)
    print(f"   Dataset      : {dataset_path}")
    print(f"   Tokenizador  : {tokenizer_path}")
    print(f"   Modelo .pth  : {checkpoint_path}")
    print("="*55)

    if not os.path.exists(checkpoint_path):
        print("   Modelo no encontrado. Iniciando entrenamiento...")
        entrenar_modelo()
        print("   Entrenamiento completado. Modelo guardado.")

    print("   Cargando modelo en memoria...")
    # cargar_modelo detecta automáticamente el mejor backend disponible:
    #   → ONNX Runtime DirectML (GPU Intel Iris Xe)  si hay .onnx
    #   → PyTorch + torch-directml                   como fallback
    app.state.model, app.state.tokenizer, app.state.backend = cargar_modelo(
        checkpoint_path, tokenizer_path
    )

    print(f"   Backend activo : {app.state.backend}")
    print("="*55 + "\n")
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


@app.get("/status")
def get_status():
    modelo_listo = os.path.exists(checkpoint_path)
    onnx_listo   = os.path.exists(os.path.splitext(checkpoint_path)[0] + ".onnx")

    total_casos = 0
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_casos = len(data)
        except Exception:
            pass

    backend = getattr(app.state, "backend", "No cargado aún")

    return {
        "status"           : "online",
        "backend"          : backend,
        "modelo_pth"       : modelo_listo,
        "modelo_onnx"      : onnx_listo,
        "casos_en_dataset" : total_casos,
        "mensaje"          : (
            f"Servidor activo. Modelo listo con {total_casos} caso(s) de entrenamiento."
            if modelo_listo else
            "Servidor activo. Usa POST /train para entrenar el modelo."
        )
    }


@app.post("/diagnose")
async def api_diagnose(
    prompt: str        = Form(..., description="Descripción del problema de red, logs o comandos a analizar."),
    image : UploadFile = File(None, description="Imagen opcional: topología, captura de Wireshark, etc.")
):
    """Diagnóstico local usando únicamente el modelo entrenado."""
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
        respuesta = diagnosticar(
            prompt,
            app.state.model,
            app.state.tokenizer,
            image_path=temp_img_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante inferencia: {e}")
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass

    return {
        "status"   : "success",
        "backend"  : app.state.backend,
        "prompt"   : prompt,
        "respuesta": respuesta
    }


@app.post("/diagnose-web")
async def api_diagnose_web(
    prompt: str        = Form(..., description="Descripción del problema de red a diagnosticar y buscar en la web."),
    image : UploadFile = File(None, description="Imagen opcional: topología o captura.")
):
    """Diagnóstico híbrido: modelo local + búsqueda web (DuckDuckGo)."""
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
        respuesta_modelo = diagnosticar(
            prompt,
            app.state.model,
            app.state.tokenizer,
            image_path=temp_img_path,
        )
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
        "backend"         : app.state.backend,
        "tiene_imagen"    : temp_img_path is not None,
        "fuente"          : "modelo_local + web",
        "prompt"          : prompt,
        "respuesta_modelo": respuesta_modelo,
        "contexto_web"    : contexto_web if contexto_web else "Sin resultados adicionales de la web."
    }


@app.post("/feedback")
async def api_feedback(
    prompt   : str = Form(..., description="El prompt o problema de red que se describió."),
    respuesta: str = Form(..., description="La respuesta correcta (Diagnóstico + Causas + Solución)."),
    titulo   : str = Form("Sin título", description="Título opcional del caso para identificarlo.")
):
    """Guarda un caso correcto en el dataset para el próximo re-entrenamiento."""
    if not prompt.strip() or not respuesta.strip():
        raise HTTPException(
            status_code=400,
            detail="El prompt y la respuesta son obligatorios y no pueden estar vacíos."
        )

    try:
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        else:
            dataset = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer dataset: {e}")

    nuevo_id   = f"caso_fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    nuevo_caso = {
        "id"     : nuevo_id,
        "titulo" : titulo.strip(),
        "prompt" : prompt.strip(),
        "target" : respuesta.strip(),
        "origen" : "feedback_usuario",
        "fecha"  : datetime.now().isoformat()
    }
    dataset.append(nuevo_caso)

    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar en dataset: {e}")

    try:
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n{prompt}\n{respuesta}\n")
    except Exception:
        pass

    return {
        "status"     : "success",
        "mensaje"    : f"Caso '{nuevo_id}' guardado. El modelo aprenderá en el próximo entrenamiento.",
        "id_caso"    : nuevo_id,
        "total_casos": len(dataset),
        "sugerencia" : "Llama a POST /train para re-entrenar el modelo con el nuevo caso."
    }


@app.post("/train")
def api_train():
    """
    Re-entrena el modelo con el dataset actual y exporta un nuevo .onnx.
    Reinicia el servidor para cargar los nuevos pesos en memoria.
    """
    try:
        entrenar_modelo()
        onnx_path = os.path.splitext(checkpoint_path)[0] + ".onnx"
        return {
            "status"    : "success",
            "mensaje"   : "Modelo re-entrenado y exportado. Reinicia el servidor para cargar los nuevos pesos.",
            "modelo_pth": checkpoint_path,
            "modelo_onnx": onnx_path
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