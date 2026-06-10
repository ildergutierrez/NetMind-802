import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from tokenizers import Tokenizer
from model_architecture import NetMindVLM

# Configurar codificación UTF-8 en Windows para evitar errores de consola
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

@torch.no_grad()
def diagnosticar(prompt_txt, image_path=None, device="cpu"):
    """
    Función de inferencia para el modelo NetMindVLM.
    Carga el modelo y genera la respuesta analítica palabra por palabra.
    """
    # 1. Definir Rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    tokenizer_path = os.path.join(current_dir, "tokenizer_redes.json")
    checkpoint_path = os.path.join(project_root, "models", "netmind_vlm.pth")
    
    if not os.path.exists(checkpoint_path):
        return "❌ Error: El modelo aún no ha terminado de entrenarse o no tiene un archivo de pesos guardado en models/netmind_vlm.pth."
        
    # 2. Cargar Tokenizador y Modelo
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = NetMindVLM(vocab_size=2000, embed_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 3. Procesar Imagen
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            image_tensor = img_transforms(img).unsqueeze(0).to(device) # Añadir dimensión de lote
        except Exception as e:
            print(f"⚠️ Error cargando imagen {image_path}: {e}. Se usará padding visual.")
            image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    else:
        image_tensor = torch.zeros(1, 3, 224, 224).to(device)
        
    # 4. Tokenizar el Prompt de Entrada
    encoding = tokenizer.encode(prompt_txt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
    
    # 5. Generación Autorregresiva (Greedy Search)
    # Iniciamos la decodificación con el token especial [CLS] (ID 2)
    target_ids = torch.tensor([[2]], dtype=torch.long).to(device)
    max_gen_len = 256
    
    for _ in range(max_gen_len):
        # Obtener predicciones para la secuencia actual del decoder
        outputs = model(image_tensor, input_ids, target_ids)
        
        # Obtener los logits del último token generado
        next_token_logits = outputs[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        
        # Si el modelo predice el token especial de parada [SEP] (ID 3), terminamos
        if next_token_id == 3:
            break
            
        # Concatenar el nuevo token generado al decoder input
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
        target_ids = torch.cat([target_ids, next_token_tensor], dim=1)
        
    # 6. Convertir IDs de tokens generados de vuelta a texto
    generated_ids = target_ids[0, 1:].tolist() # Omitir el primer token [CLS]
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔎 Probando inferencia local (usando {device})...")
    
    prompt_test = "El puerto GigabitEthernet1/0/5 del switch Cisco de distribución se apagó repentinamente. En la consola aparece la alerta de seguridad 'port-security violation'. Un usuario intentó conectar una laptop de auditoría directamente en la toma de red. ¿Cómo se diagnostica, cuál es la causa y cómo se soluciona en Cisco IOS?"
    
    respuesta = diagnosticar(prompt_test, device=device)
    print("\n--- Respuesta del Modelo ---")
    print(respuesta)
    print("----------------------------")
