import os
import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

HF_REPO_ID      = "ildergutierrez12/netmind-802"
HF_MODEL_FILE   = "netmind_vlm.pth"
MAX_GEN_LEN     = 300
TEMPERATURE     = 0.7
TOP_K           = 50
REPETITION_PEN  = 1.3
TOKEN_CLS       = 2
TOKEN_SEP       = 3

_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _preprocesar_imagen(image_path):
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMG_MEAN) / _IMG_STD
    return arr.transpose(2, 0, 1)[np.newaxis]

def _imagen_padding():
    return np.zeros((1, 3, 224, 224), dtype=np.float32)


def cargar_modelo(checkpoint_path, tokenizer_path, device=None):
    """
    Descarga netmind_vlm.pth desde HuggingFace Hub si no existe localmente,
    luego carga el modelo PyTorch custom (NetMindVLM).
    """
    from tokenizers import Tokenizer

    # ── Descargar .pth desde HuggingFace si no existe ──────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"   Descargando modelo desde {HF_REPO_ID} ...")
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            tmp = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILE)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            shutil.copy2(tmp, checkpoint_path)
            print(f"   ✔ Modelo descargado en: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"No se pudo descargar el modelo desde HuggingFace: {e}")
    else:
        print(f"   Modelo encontrado localmente: {checkpoint_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    import torch
    from model_architecture import NetMindVLM

    try:
        import torch_directml
        pt_device = torch_directml.device()
        backend   = "PyTorch — GPU Intel Iris Xe (DirectML)"
    except ImportError:
        pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backend   = f"PyTorch — {str(pt_device).upper()}"

    model = NetMindVLM(vocab_size=2000, embed_dim=256)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(pt_device)
    model.eval()
    print(f"   Backend activo: {backend}")
    return model, tokenizer, backend


def _sample_token_torch(logits_1d, generated):
    import torch
    logits = logits_1d.float().clone()
    for tok_id in set(generated):
        logits[tok_id] = logits[tok_id] / REPETITION_PEN if logits[tok_id] > 0 else logits[tok_id] * REPETITION_PEN
    logits = logits / max(TEMPERATURE, 1e-8)
    if TOP_K > 0:
        valores, _ = torch.topk(logits, TOP_K)
        logits[logits < valores[-1]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def diagnosticar(prompt_txt, model, tokenizer, image_path=None, device=None):
    import torch
    try:
        pt_device = next(model.parameters()).device
    except StopIteration:
        pt_device = torch.device("cpu")

    if image_path and os.path.exists(image_path):
        try:
            img_np = _preprocesar_imagen(image_path)
            image_tensor = torch.from_numpy(img_np).to(pt_device)
        except Exception:
            image_tensor = torch.zeros(1, 3, 224, 224, device=pt_device)
    else:
        image_tensor = torch.zeros(1, 3, 224, 224, device=pt_device)

    encoding  = tokenizer.encode(prompt_txt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=pt_device)

    target_ids = torch.tensor([[TOKEN_CLS]], dtype=torch.long, device=pt_device)
    generated  = []

    with torch.no_grad():
        for _ in range(MAX_GEN_LEN):
            outputs   = model(image_tensor, input_ids, target_ids)
            next_tok  = _sample_token_torch(outputs[0, -1, :], generated)
            if next_tok == TOKEN_SEP:
                break
            if len(generated) >= 5 and all(t == next_tok for t in generated[-5:]):
                break
            target_ids = torch.cat(
                [target_ids, torch.tensor([[next_tok]], dtype=torch.long, device=pt_device)], dim=1
            )
            generated.append(next_tok)

    import re
    texto = tokenizer.decode(generated)
    texto = texto.replace("Ġ", " ").replace("Ċ", "\n")
    texto = re.sub(r" {2,}", " ", texto).strip()
    return texto or "No se pudo generar respuesta."


if __name__ == "__main__":
    current_dir     = os.path.dirname(os.path.abspath(__file__))
    project_root    = os.path.dirname(current_dir)
    tokenizer_path  = os.path.join(current_dir, "tokenizer_redes.json")
    checkpoint_path = os.path.join(project_root, "models", "netmind_vlm.pth")

    print("🔎 Cargando modelo desde HuggingFace Hub...")
    model, tokenizer, backend = cargar_modelo(checkpoint_path, tokenizer_path)
    print(f"   Backend: {backend}\n")

    prompt = "El puerto GigabitEthernet1/0/5 del switch Cisco se apagó. Aparece 'port-security violation'. ¿Cómo se soluciona?"
    print(f"Prompt: {prompt}")
    print("Respuesta:")
    print(diagnosticar(prompt, model, tokenizer))