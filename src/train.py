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


def detectar_device():
    """
    Detecta el mejor dispositivo disponible en el siguiente orden de prioridad:
      1. GPU Intel/AMD vía DirectML (torch-directml) — ideal para i5-13xxx con Iris Xe
      2. CUDA (si existiera una GPU NVIDIA)
      3. CPU como último recurso
    Retorna (device, label).
    """
    try:
        import torch_directml
        dml_device = torch_directml.device()
        return dml_device, "GPU Intel Iris Xe (DirectML)"
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda"), f"GPU CUDA ({torch.cuda.get_device_name(0)})"

    return torch.device("cpu"), "CPU"


def exportar_onnx(model, models_dir, device):
    """
    Exporta el modelo entrenado a formato ONNX para inferencia con ONNX Runtime DirectML.

    El modelo se mueve a CPU para la exportación porque torch.onnx.export
    no es compatible con dispositivos DirectML. El .onnx resultante se puede
    ejecutar en cualquier backend (DirectML, CPU, CUDA).

    Entradas del modelo:
        images      : (1, 3, 224, 224)  — tensor visual
        input_ids   : (1, seq_len)       — tokens del prompt
        target_ids  : (1, tgt_len)       — tokens de decodificación
    Salida:
        logits      : (1, tgt_len, vocab_size)
    """
    onnx_path = os.path.join(models_dir, "netmind_vlm.onnx")
    print(f"\n📦 Exportando modelo a ONNX...")
    print(f"   Destino: {onnx_path}")

    # Mover el modelo a CPU para la exportación (requisito de torch.onnx.export)
    model_cpu = model.to("cpu")
    model_cpu.eval()

    # Entradas de ejemplo con dimensiones representativas
    dummy_images     = torch.zeros(1, 3, 224, 224)
    dummy_input_ids  = torch.zeros(1, 32, dtype=torch.long)
    dummy_target_ids = torch.zeros(1, 16, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            (dummy_images, dummy_input_ids, dummy_target_ids),
            onnx_path,
            opset_version=17,
            input_names=["images", "input_ids", "target_ids"],
            output_names=["logits"],
            dynamic_axes={
                "images"     : {0: "batch"},
                "input_ids"  : {0: "batch", 1: "src_len"},
                "target_ids" : {0: "batch", 1: "tgt_len"},
                "logits"     : {0: "batch", 1: "tgt_len"},
            },
            do_constant_folding=True,
            export_params=True,
        )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   ✅ ONNX exportado correctamente ({size_mb:.1f} MB)")
    print(f"   ℹ️  Usa ONNX Runtime DirectML para inferencia acelerada con la GPU Intel.")

    # Devolver el modelo al device original
    model.to(device)
    return onnx_path


def entrenar_modelo():
    # 1. Definir Rutas
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    json_path       = os.path.join(project_root, "data", "dataset_redes.json")
    tokenizer_path  = os.path.join(current_dir, "tokenizer_redes.json")
    models_dir      = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "netmind_vlm.pth")

    # 2. Configurar Dispositivo
    device, device_label = detectar_device()
    print(f"🖥️  Dispositivo de entrenamiento: {device_label}")

    # 3. Parámetros de Entrenamiento
    batch_size    = 2
    max_len       = 256
    epochs        = 100   # Dataset pequeño → más épocas para memorizar los casos
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
    model = NetMindVLM(vocab_size=2000, embed_dim=256).to(device)

    # 6. Definir Pérdida y Optimizador
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore_index=0 → ignora [PAD]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("⏳ Iniciando bucle de entrenamiento...")
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for batch in dataloader:
            images     = batch["image"].to(device)
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # Decodificación autorregresiva:
            #   decoder_input   → target sin el último token (contexto que ve el decoder)
            #   decoder_targets → target sin el primer token (siguiente token a predecir)
            decoder_input   = target_ids[:, :-1]
            decoder_targets = target_ids[:, 1:]

            outputs = model(images, input_ids, decoder_input)

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                decoder_targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"   Epoch {epoch:03d}/{epochs:03d} | Pérdida (Loss): {avg_loss:.6f}")

    # 7. Guardar Pesos PyTorch (.pth)
    print(f"\n💾 Guardando pesos del modelo en:\n   {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)
    print("✅ Pesos guardados correctamente.")

    # 8. Exportar a ONNX para inferencia con ONNX Runtime DirectML
    exportar_onnx(model, models_dir, device)

    print("\n🎉 ¡Entrenamiento completado!")
    print("   Archivos generados:")
    print(f"   • {checkpoint_path}  (pesos PyTorch)")
    print(f"   • {os.path.join(models_dir, 'netmind_vlm.onnx')}  (modelo ONNX para ONNX Runtime)")


if __name__ == "__main__":
    entrenar_modelo()