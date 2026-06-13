import os
import sys
import torch
import torch.nn as nn
from dataset_loader import crear_dataloader
from model_architecture import NetMindVLM, VOCAB_SIZE

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def detectar_device():
    try:
        import torch_directml
        return torch_directml.device(), "GPU Intel Iris Xe (DirectML)"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), f"GPU CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"


def exportar_onnx(model, models_dir, device):
    onnx_path = os.path.join(models_dir, "netmind_vlm.onnx")
    print(f"\n📦 Exportando a ONNX → {onnx_path}")
    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy_images     = torch.zeros(1, 3, 224, 224)
    dummy_input_ids  = torch.zeros(1, 32, dtype=torch.long)
    dummy_target_ids = torch.zeros(1, 16, dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            (dummy_images, dummy_input_ids, dummy_target_ids),
            onnx_path, opset_version=17,
            input_names=["images", "input_ids", "target_ids"],
            output_names=["logits"],
            dynamic_axes={
                "images":     {0: "batch"},
                "input_ids":  {0: "batch", 1: "src_len"},
                "target_ids": {0: "batch", 1: "tgt_len"},
                "logits":     {0: "batch", 1: "tgt_len"},
            },
        )
    print(f"   ✅ ONNX exportado ({os.path.getsize(onnx_path)/1e6:.1f} MB)")
    model.to(device)
    return onnx_path


def entrenar_modelo():
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    json_path       = os.path.join(project_root, "data", "dataset_redes.json")
    tokenizer_path  = os.path.join(current_dir, "tokenizer_redes.json")
    models_dir      = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "netmind_vlm.pth")

    device, device_label = detectar_device()
    print(f"🖥️  Device: {device_label}")

    print("📖 Cargando dataset...")
    dataloader = crear_dataloader(
        json_path=json_path, tokenizer_path=tokenizer_path,
        batch_size=4, shuffle=True, max_len=256
    )
    print(f"   {len(dataloader)} batches")

    print(f"🤖 Inicializando NetMindVLM (vocab={VOCAB_SIZE})...")
    model     = NetMindVLM(vocab_size=VOCAB_SIZE, embed_dim=256).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("⏳ Entrenando 100 épocas...\n")
    model.train()
    for epoch in range(1, 101):
        epoch_loss = 0.0
        for batch in dataloader:
            images     = batch["image"].to(device)
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            dec_in  = target_ids[:, :-1]
            dec_tgt = target_ids[:, 1:]

            out  = model(images, input_ids, dec_in)
            loss = criterion(out.reshape(-1, out.size(-1)), dec_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}/100 | Loss: {epoch_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Modelo guardado: {checkpoint_path}")
    exportar_onnx(model, models_dir, device)
    print("\n🎉 Entrenamiento completado.")


if __name__ == "__main__":
    entrenar_modelo()