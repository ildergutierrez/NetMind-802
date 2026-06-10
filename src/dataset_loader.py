import os
import json
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tokenizers import Tokenizer

# Configurar codificación UTF-8 para evitar errores con emojis en consolas de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class DatasetRedes(Dataset):
    """
    Dataset personalizado para el modelo NetMind-802.
    Procesa entradas multimodales (Texto de diagnóstico + Imagen opcional de topología)
    y retorna tensores de PyTorch listos para el entrenamiento.
    """
    def __init__(self, json_path, tokenizer_path, img_dir=None, max_len=256):
        self.max_len = max_len
        self.img_dir = img_dir
        
        # Cargar base de conocimiento JSON
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No se encontró el dataset JSON en {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Cargar tokenizador BPE entrenado
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"No se encontró el tokenizador JSON en {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Configurar padding y truncamiento automático en el tokenizador
        # Usamos [PAD] que registramos como token especial (ID 0)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)
        
        # Definir transformaciones de imágenes para el codificador visual (ResNet/ViT)
        self.img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # Valores estándar de ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_txt = item["prompt"]
        target_txt = item["target"]
        
        # 1. PROCESAR IMAGEN (Si tiene una ruta asignada y existe, la carga. Si no, crea un tensor vacío)
        image_tensor = None
        has_image = False
        
        if "image_path" in item and item["image_path"]:
            # Resolver la ruta relativa al directorio de imágenes
            img_path = item["image_path"]
            if self.img_dir:
                img_path = os.path.join(self.img_dir, img_path)
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    image_tensor = self.img_transforms(img)
                    has_image = True
                except Exception as e:
                    print(f"⚠️ Error al abrir la imagen {img_path}: {e}")
                    
        # Si no hay imagen o falló la carga, creamos un tensor de ceros como relleno (Padding visual)
        if not has_image:
            image_tensor = torch.zeros(3, 224, 224)
            
        # 2. PROCESAR TEXTO (Tokenización)
        # Codificar prompt (entrada) y target (salida esperada)
        encoding_prompt = self.tokenizer.encode(prompt_txt)
        encoding_target = self.tokenizer.encode(target_txt)
        
        # Convertir a tensores de PyTorch (tipo entero largo para IDs de tokens)
        input_ids = torch.tensor(encoding_prompt.ids, dtype=torch.long)
        attention_mask = torch.tensor(encoding_prompt.attention_mask, dtype=torch.long)
        
        target_ids = torch.tensor(encoding_target.ids, dtype=torch.long)
        target_mask = torch.tensor(encoding_target.attention_mask, dtype=torch.long)
        
        return {
            "id": item["id"],
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "has_image": torch.tensor(1.0 if has_image else 0.0, dtype=torch.float)
        }

def crear_dataloader(json_path, tokenizer_path, img_dir=None, batch_size=2, shuffle=True, max_len=256):
    """
    Función helper para instanciar el dataset y crear el DataLoader de PyTorch.
    """
    dataset = DatasetRedes(
        json_path=json_path,
        tokenizer_path=tokenizer_path,
        img_dir=img_dir,
        max_len=max_len
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 # Mantener en 0 para evitar problemas de multiprocessing en Windows
    )
    
    return loader

# Código de prueba para verificar que el loader funciona de forma autónoma
if __name__ == "__main__":
    # Obtener rutas absolutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    json_path = os.path.join(project_root, "data", "dataset_redes.json")
    tokenizer_path = os.path.join(current_dir, "tokenizer_redes.json")
    
    try:
        loader = crear_dataloader(json_path, tokenizer_path, batch_size=2, shuffle=True)
        print("🔍 Cargando un lote de prueba...")
        for batch in loader:
            print("\n✅ Lote cargado exitosamente:")
            print(f"  IDs de casos: {batch['id']}")
            print(f"  Shape de imágenes: {batch['image'].shape}") # Debe ser [batch_size, 3, 224, 224]
            print(f"  Shape de input_ids: {batch['input_ids'].shape}") # Debe ser [batch_size, max_len]
            print(f"  Shape de target_ids: {batch['target_ids'].shape}") # Debe ser [batch_size, max_len]
            print(f"  ¿Tiene imagen real?: {batch['has_image'].tolist()}")
            break
    except Exception as e:
        print(f"❌ Error al probar el DataLoader: {e}")
