import os
import sys
import torch
import torch.nn as nn
from dataset_loader import crear_dataloader
from model_architecture import NetMindVLM

# Configurar codificación UTF-8 en Windows para evitar errores de consola
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def entrenar_modelo():
    # 1. Definir Rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    json_path = os.path.join(project_root, "data", "dataset_redes.json")
    tokenizer_path = os.path.join(current_dir, "tokenizer_redes.json")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "netmind_vlm.pth")
    
    # 2. Configurar Dispositivo (GPU CUDA si está disponible, si no CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo de entrenamiento seleccionado: {device}")
    
    # 3. Parámetros de Entrenamiento
    batch_size = 2
    max_len = 256
    epochs = 100 # Como es una base de conocimiento pequeña, entrenamos más épocas para memorizar/sobreajustar los casos
    learning_rate = 1e-4
    
    # 4. Cargar DataLoader
    print("📖 Cargando base de datos...")
    dataloader = crear_dataloader(
        json_path=json_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        shuffle=True,
        max_len=max_len
    )
    
    # 5. Instanciar Modelo
    print("🤖 Inicializando modelo NetMindVLM...")
    # El tamaño del vocabulario debe coincidir con el del tokenizador (2000)
    model = NetMindVLM(vocab_size=2000, embed_dim=256).to(device)
    
    # 6. Definir Pérdida y Optimizado
    # ignore_index=0 hace que el cálculo de pérdida ignore el token especial [PAD] (ID 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("⏳ Iniciando bucle de entrenamiento...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # En la decodificación autorregresiva de texto:
            # - La entrada del decoder es target_ids eliminando el último token (para predecir el siguiente)
            # - La etiqueta real a predecir es target_ids omitiendo el primer token (el siguiente token)
            decoder_input = target_ids[:, :-1]
            decoder_targets = target_ids[:, 1:]
            
            # Pasada hacia adelante (forward pass)
            outputs = model(images, input_ids, decoder_input)
            
            # Formatear salidas y objetivos para calcular pérdida de Entropía Cruzada
            # outputs shape: [batch * (seq_len-1), vocab_size]
            # decoder_targets shape: [batch * (seq_len-1)]
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)), 
                decoder_targets.reshape(-1)
            )
            
            # Optimización (Retropropagación)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Imprimir progreso cada 10 épocas para no saturar la consola
        if epoch == 1 or epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"   Epoch {epoch:03d}/{epochs:03d} | Pérdida (Loss): {avg_loss:.6f}")
            
    # 7. Guardar Pesos Entrenados
    print(f"💾 Guardando pesos del modelo en:\n   {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)
    print("🎉 ¡Entrenamiento completado y modelo listo!")

if __name__ == "__main__":
    entrenar_modelo()
