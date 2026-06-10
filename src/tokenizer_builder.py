import os
import sys

# Configurar codificación UTF-8 en Windows para evitar errores de impresión
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def entrenar_tokenizador():
    # Obtener la ruta absoluta del corpus independientemente de dónde se ejecute el script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "corpus_redes.txt")
    output_path = os.path.join(current_dir, "tokenizer_redes.json")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró el archivo de corpus en {data_path}")
        return
        
    print(f"📖 Cargando corpus desde: {data_path}")
    print("⏳ Iniciando el entrenamiento del tokenizador BPE...")
    
    # Inicializar el modelo BPE (Byte Pair Encoding)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Pre-tokenizar separando palabras por espacios en blanco
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configurar el entrenador
    # Usamos un vocabulario pequeño (2000 tokens) porque nuestro corpus de redes es especializado y compacto
    trainer = BpeTrainer(
        vocab_size=2000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Entrenar el modelo con el corpus
    tokenizer.train([data_path], trainer)
    
    # Guardar el archivo de configuración del tokenizador
    tokenizer.save(output_path)
    print(f"✅ Tokenizador entrenado con éxito y guardado en:\n   {output_path}")

    # Probar el tokenizador
    print("\n🔍 Realizando prueba de tokenización:")
    frase_prueba = "switchport port-security sticky en interface GigabitEthernet1/0/5 de Cisco"
    resultado = tokenizer.encode(frase_prueba)
    
    print(f"  Texto original: '{frase_prueba}'")
    print(f"  Tokens: {resultado.tokens}")
    print(f"  IDs de tokens: {resultado.ids}")

if __name__ == "__main__":
    entrenar_tokenizador()
