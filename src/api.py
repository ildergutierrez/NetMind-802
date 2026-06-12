"""
=======================================================
 api.py — NetMind-802 para HuggingFace Spaces
=======================================================
 Cambios vs versión local:
   - Descarga netmind_vlm.pth desde HuggingFace Hub
     al arrancar si no existe en /tmp/models/
   - Rutas adaptadas al sistema de archivos de Spaces
   - Sin torch-directml (Spaces usa GPU NVIDIA o CPU)
   - POST /train deshabilitado (ZeroGPU no alcanza)
   - Escucha en 0.0.0.0:7860 (puerto estándar de Spaces)
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
# RUTAS EN HUGGINGFACE SPACES
#   /app/        → código del repo (src/)
#   /tmp/models/ → modelo descargado del Hub (efímero)
#   /tmp/data/   → dataset + feedback (efímero por sesión)
# -------------------------------------------------------
current_dir     = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = "/tmp/models/netmind_vlm.pth"
tokenizer_path  = os.path.join(current_dir, "tokenizer_redes.json")
dataset_path    = "/tmp/data/dataset_redes.json"
corpus_path     = "/tmp/data/corpus_redes.txt"

# Repo del Hub donde está subido el .pth
#   → cámbialo si tu usuario o repo name es distinto
HF_REPO_ID  = "ildergutierrez12/netmind-802"
HF_FILENAME = "netmind_vlm.pth"

os.makedirs("/tmp/models", exist_ok=True)
os.makedirs("/tmp/data",   exist_ok=True)

# -------------------------------------------------------
# IMPORTAR LIBRERÍAS DEL SERVIDOR
# -------------------------------------------------------
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------
# IMPORTAR MÓDULOS INTERNOS
# -------------------------------------------------------
from test_inference import cargar_modelo, diagnosticar
from web_search import construir_contexto_web


# -------------------------------------------------------
# DESCARGA DEL MODELO DESDE EL HUB
# -------------------------------------------------------
def descargar_modelo_si_falta():
    """
    Descarga netmind_vlm.pth desde HuggingFace Hub a /tmp/models/
    solo si no existe ya (para no re-descargar en cada request).
    """
    if os.path.exists(checkpoint_path):
        print(f"   Modelo ya en caché: {checkpoint_path}")
        return

    print(f"   Descargando modelo desde Hub: {HF_REPO_ID}/{HF_FILENAME} ...")
    try:
        from huggingface_hub import hf_hub_download
        ruta = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir="/tmp/models",
        )
        print(f"   ✅ Modelo descargado en: {ruta}")
    except Exception as e:
        raise RuntimeError(
            f"No se pudo descargar el modelo desde '{HF_REPO_ID}'.\n"
            f"Verifica que el repo existe y el archivo es '{HF_FILENAME}'.\n"
            f"Error: {e}"
        )


# -------------------------------------------------------
# COPIAR DATASET DEL REPO A /tmp/data/ (si existe)
# -------------------------------------------------------
def inicializar_dataset():
    """
    Copia dataset_redes.json del repo a /tmp/data/ para que
    POST /feedback pueda escribir sin tocar el repo de solo lectura.
    """
    repo_dataset = os.path.join(os.path.dirname(current_dir), "data", "dataset_redes.json")
    if not os.path.exists(dataset_path) and os.path.exists(repo_dataset):
        shutil.copy(repo_dataset, dataset_path)
        print(f"   Dataset copiado a {dataset_path}")


# -------------------------------------------------------
# LIFESPAN — arranque del servidor
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*55)
    print("   NetMind-802 — HuggingFace Spaces")
    print("="*55)
    print(f"   Hub repo     : {HF_REPO_ID}")
    print(f"   Modelo .pth  : {checkpoint_path}")
    print(f"   Tokenizador  : {tokenizer_path}")
    print("="*55)

    # 1. Descargar .pth desde el Hub si hace falta
    descargar_modelo_si_falta()

    # 2. Preparar dataset en /tmp
    inicializar_dataset()

    # 3. Cargar modelo en memoria
    #    En Spaces con ZeroGPU → CUDA disponible brevemente
    #    Sin ZeroGPU            → CPU automáticamente
    print("   Cargando modelo en memoria...")
    app.state.model, app.state.tokenizer, app.state.backend = cargar_modelo(
        checkpoint_path, tokenizer_path
    )
    print(f"   Backend activo : {app.state.backend}")
    print("="*55 + "\n")
    yield
    print("\n   Servidor NetMind-802 detenido.")


# -------------------------------------------------------
# APP FASTAPI
# -------------------------------------------------------
app = FastAPI(
    title="NetMind-802 API",
    description=(
        "IA especializada en redes cableadas e inalámbricas (IEEE 802), "
        "infraestructura de red, telemática y seguridad multiplataforma. "
        "Compatible con Cisco IOS, Windows PowerShell y Linux Bash."
    ),
    version="1.0.0",
    lifespan=lifespan
)

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
        "hub"        : f"https://huggingface.co/{HF_REPO_ID}",
        "endpoints"  : {
            "GET  /status"      : "Estado del servidor y del modelo",
            "POST /diagnose"    : "Diagnosticar un problema de red (texto + imagen opcional)",
            "POST /diagnose-web": "Diagnosticar + búsqueda web (DuckDuckGo)",
            "POST /feedback"    : "Enviar un caso correcto para futuros entrenamientos",
        }
    }


@app.get("/status")
def get_status():
    modelo_listo = os.path.exists(checkpoint_path)
    backend      = getattr(app.state, "backend", "No cargado aún")

    total_casos = 0
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                total_casos = len(json.load(f))
        except Exception:
            pass

    return {
        "status"          : "online",
        "backend"         : backend,
        "modelo_pth"      : modelo_listo,
        "hub_repo"        : HF_REPO_ID,
        "casos_en_dataset": total_casos,
        "mensaje"         : (
            f"Servidor activo. Modelo cargado desde HuggingFace Hub."
            if modelo_listo else
            "Modelo no encontrado. Verifica que el .pth está en el Hub."
        )
    }


@app.post("/diagnose")
async def api_diagnose(
    prompt: str        = Form(..., description="Descripción del problema de red, logs o comandos."),
    image : UploadFile = File(None, description="Imagen opcional: topología, captura de Wireshark, etc.")
):
    """Diagnóstico usando el modelo cargado desde HuggingFace Hub."""
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
    prompt: str        = Form(..., description="Problema de red a diagnosticar y buscar en la web."),
    image : UploadFile = File(None, description="Imagen opcional.")
):
    """Diagnóstico híbrido: modelo + búsqueda web (DuckDuckGo)."""
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
        "fuente"          : "modelo_hub + web",
        "prompt"          : prompt,
        "respuesta_modelo": respuesta_modelo,
        "contexto_web"    : contexto_web or "Sin resultados adicionales de la web."
    }


@app.get("/dataset")
def get_dataset():
    """Retorna todos los casos del dataset acumulado (para descargar desde Colab)."""
    if not os.path.exists(dataset_path):
        return []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer dataset: {e}")


@app.post("/feedback")
async def api_feedback(
    prompt   : str = Form(..., description="El problema de red descrito."),
    respuesta: str = Form(..., description="La respuesta correcta (Diagnóstico + Causas + Solución)."),
    titulo   : str = Form("Sin título", description="Título opcional del caso.")
):
    """
    Guarda un caso correcto en /tmp/data/dataset_redes.json.
    Nota: en Spaces los datos de /tmp son efímeros (se pierden al reiniciar).
    Para persistencia real, conecta un HuggingFace Dataset vía API.
    """
    if not prompt.strip() or not respuesta.strip():
        raise HTTPException(status_code=400, detail="Prompt y respuesta son obligatorios.")

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
        "id"    : nuevo_id,
        "titulo": titulo.strip(),
        "prompt": prompt.strip(),
        "target": respuesta.strip(),
        "origen": "feedback_usuario",
        "fecha" : datetime.now().isoformat()
    }
    dataset.append(nuevo_caso)

    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n{prompt}\n{respuesta}\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar: {e}")

    return {
        "status"     : "success",
        "mensaje"    : f"Caso '{nuevo_id}' guardado en sesión actual.",
        "advertencia": "Los datos en /tmp son efímeros en Spaces. Para persistirlos, descárgalos y vuelve a subir el dataset al Hub.",
        "id_caso"    : nuevo_id,
        "total_casos": len(dataset)
    }


# =======================================================
# PUNTO DE ENTRADA — puerto 7860 (estándar de HF Spaces)
# =======================================================
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )