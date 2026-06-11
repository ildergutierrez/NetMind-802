import os
import sys
import numpy as np

# Configurar codificación UTF-8 en Windows para evitar errores de consola
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Transformaciones de imagen (sin torchvision, solo numpy/PIL)
# ---------------------------------------------------------------------------
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocesar_imagen(image_path: str) -> np.ndarray:
    """
    Carga una imagen, la redimensiona a 224×224, normaliza con ImageNet
    y retorna un array float32 de shape (1, 3, 224, 224).
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0        # (224, 224, 3)  → [0, 1]
    arr = (arr - _IMG_MEAN) / _IMG_STD                   # Normalización ImageNet
    arr = arr.transpose(2, 0, 1)[np.newaxis]             # (1, 3, 224, 224)
    return arr


def _imagen_padding() -> np.ndarray:
    """Retorna un tensor de imagen en negro (sin imagen disponible)."""
    return np.zeros((1, 3, 224, 224), dtype=np.float32)


# ---------------------------------------------------------------------------
# Carga del modelo: ONNX Runtime DirectML  (GPU Intel Iris Xe)
# ---------------------------------------------------------------------------

def cargar_modelo(checkpoint_path, tokenizer_path, device=None):
    """
    Carga el tokenizador y el modelo para inferencia.

    Estrategia de backends (en orden de prioridad):
      1. ONNX Runtime con DirectMLExecutionProvider  → GPU Intel Iris Xe
      2. ONNX Runtime con CPUExecutionProvider       → CPU (fallback)
      3. PyTorch + torch-directml                    → fallback si no hay .onnx

    Parámetros
    ----------
    checkpoint_path : str
        Ruta al archivo .pth (PyTorch). Se deduce la ruta .onnx automáticamente.
    tokenizer_path  : str
        Ruta al archivo tokenizer_redes.json.
    device          : ignorado — se selecciona automáticamente.

    Retorna
    -------
    (session_o_model, tokenizer, backend_label)
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Ruta al .onnx (mismo directorio que el .pth)
    onnx_path = os.path.splitext(checkpoint_path)[0] + ".onnx"

    # ── Intento 1: ONNX Runtime DirectML ────────────────────────────────────
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort

            disponibles = ort.get_available_providers()

            if "DmlExecutionProvider" in disponibles:
                session = ort.InferenceSession(
                    onnx_path,
                    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
                )
                backend = "ONNX Runtime — GPU Intel Iris Xe (DirectML)"
            else:
                session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                backend = "ONNX Runtime — CPU (DirectML no disponible)"

            print(f"   Modelo cargado con {backend}.")
            return session, tokenizer, backend

        except ImportError:
            print("   ⚠️  onnxruntime no instalado. Intentando PyTorch…")
        except Exception as e:
            print(f"   ⚠️  Error al cargar ONNX ({e}). Intentando PyTorch…")

    # ── Intento 2 / Fallback: PyTorch + torch-directml ──────────────────────
    import torch
    from model_architecture import NetMindVLM

    try:
        import torch_directml
        pt_device = torch_directml.device()
        backend   = "PyTorch — GPU Intel Iris Xe (DirectML)"
    except ImportError:
        pt_device = torch.device("cpu")
        backend   = "PyTorch — CPU"

    model = NetMindVLM(vocab_size=2000, embed_dim=256)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(pt_device)
    model.eval()
    print(f"   Modelo cargado con {backend}.")
    return model, tokenizer, backend


# ---------------------------------------------------------------------------
# Inferencia autorregresiva
# ---------------------------------------------------------------------------

def diagnosticar(prompt_txt, model, tokenizer, image_path=None, device=None):
    """
    Genera una respuesta autorregresiva token a token.

    Detecta automáticamente si `model` es una sesión ONNX Runtime
    o un módulo PyTorch y aplica la lógica correspondiente.

    Parámetros
    ----------
    prompt_txt  : str    — descripción del problema de red
    model       : InferenceSession | NetMindVLM
    tokenizer   : Tokenizer (HuggingFace tokenizers)
    image_path  : str | None
    device      : ignorado — se determina internamente
    """
    import onnxruntime as _ort_check  # noqa: F401 — sólo para ver si ORT está disponible

    try:
        import onnxruntime as ort
        _es_onnx = isinstance(model, ort.InferenceSession)
    except ImportError:
        _es_onnx = False

    if _es_onnx:
        return _diagnosticar_onnx(prompt_txt, model, tokenizer, image_path)
    else:
        return _diagnosticar_pytorch(prompt_txt, model, tokenizer, image_path)


def _diagnosticar_onnx(prompt_txt, session, tokenizer, image_path=None):
    """Inferencia con ONNX Runtime (DirectML o CPU)."""
    # 1. Imagen
    if image_path and os.path.exists(image_path):
        try:
            image_np = _preprocesar_imagen(image_path)
        except Exception as e:
            print(f"   ⚠️  Error cargando imagen: {e}. Usando padding.")
            image_np = _imagen_padding()
    else:
        image_np = _imagen_padding()

    # 2. Tokenizar prompt
    encoding  = tokenizer.encode(prompt_txt)
    input_ids = np.array([encoding.ids], dtype=np.int64)

    # 3. Generación autorregresiva — token [CLS]=2 como inicio
    target_ids  = np.array([[2]], dtype=np.int64)
    max_gen_len = 256

    for _ in range(max_gen_len):
        logits = session.run(
            ["logits"],
            {
                "images"    : image_np,
                "input_ids" : input_ids,
                "target_ids": target_ids,
            }
        )[0]                                       # (1, tgt_len, vocab_size)

        next_token_id = int(np.argmax(logits[0, -1, :]))

        if next_token_id == 3:                     # [SEP] → fin de secuencia
            break

        target_ids = np.concatenate(
            [target_ids, [[next_token_id]]], axis=1
        )

    generated_ids  = target_ids[0, 1:].tolist()   # Omitir [CLS]
    return tokenizer.decode(generated_ids)


def _diagnosticar_pytorch(prompt_txt, model, tokenizer, image_path=None):
    """Inferencia con PyTorch + torch-directml."""
    import torch

    # Determinar device desde el modelo
    try:
        pt_device = next(model.parameters()).device
    except StopIteration:
        pt_device = torch.device("cpu")

    # 1. Imagen
    if image_path and os.path.exists(image_path):
        try:
            img_np = _preprocesar_imagen(image_path)
            image_tensor = torch.from_numpy(img_np).to(pt_device)
        except Exception as e:
            print(f"   ⚠️  Error cargando imagen: {e}. Usando padding.")
            image_tensor = torch.zeros(1, 3, 224, 224).to(pt_device)
    else:
        image_tensor = torch.zeros(1, 3, 224, 224).to(pt_device)

    # 2. Tokenizar prompt
    encoding = tokenizer.encode(prompt_txt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(pt_device)

    # 3. Generación autorregresiva
    target_ids  = torch.tensor([[2]], dtype=torch.long).to(pt_device)
    max_gen_len = 256

    with torch.no_grad():
        for _ in range(max_gen_len):
            outputs        = model(image_tensor, input_ids, target_ids)
            next_token_id  = torch.argmax(outputs[0, -1, :]).item()

            if next_token_id == 3:                 # [SEP]
                break

            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(pt_device)
            target_ids        = torch.cat([target_ids, next_token_tensor], dim=1)

    generated_ids  = target_ids[0, 1:].tolist()
    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# Punto de entrada para pruebas locales
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    current_dir     = os.path.dirname(os.path.abspath(__file__))
    project_root    = os.path.dirname(current_dir)
    tokenizer_path  = os.path.join(current_dir, "tokenizer_redes.json")
    checkpoint_path = os.path.join(project_root, "models", "netmind_vlm.pth")

    if not os.path.exists(checkpoint_path):
        print("❌ Error: Modelo no encontrado en models/netmind_vlm.pth")
        print("   Ejecuta train.py primero para entrenar y exportar el modelo.")
        sys.exit(1)

    print("🔎 Cargando modelo para prueba de inferencia...")
    model, tokenizer, backend = cargar_modelo(checkpoint_path, tokenizer_path)
    print(f"   Backend activo: {backend}")

    prompt_test = (
        "El puerto GigabitEthernet1/0/5 del switch Cisco de distribución se apagó repentinamente. "
        "En la consola aparece la alerta de seguridad 'port-security violation'. "
        "Un usuario intentó conectar una laptop de auditoría directamente en la toma de red. "
        "¿Cómo se diagnostica, cuál es la causa y cómo se soluciona en Cisco IOS?"
    )

    print("\n⏳ Generando respuesta...\n")
    respuesta = diagnosticar(prompt_test, model, tokenizer)
    print("--- Respuesta del Modelo ---")
    print(respuesta)
    print("----------------------------")