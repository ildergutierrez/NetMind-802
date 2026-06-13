import torch
import torch.nn as nn
import math
import sys

# Configurar codificación UTF-8 en Windows para evitar errores de consola
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class EncoderVisual(nn.Module):
    """
    Extractor de características de imágenes desde cero (Offline-Friendly).
    Recibe un tensor de imagen (batch, 3, 224, 224) y retorna un vector de características (batch, embed_dim).
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # Bloque 1: 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 2: 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 3: 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Reducción global
            nn.AdaptiveAvgPool2d((1, 1)) # Extrae 1 valor por cada uno de los 64 canales -> (batch, 64, 1, 1)
        )
        self.proyeccion = nn.Linear(64, embed_dim) # Proyecta al tamaño de embedding del texto

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Aplanar a (batch, 64)
        return self.proyeccion(x) # (batch, embed_dim)


class CodificadorPosicional(nn.Module):
    """
    Añade información posicional a los embeddings de texto (soportando relaciones de secuencia).
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class NetMindVLM(nn.Module):
    """
    Visual Language Model (VLM) pequeño y específico para redes.
    Arquitectura: Encoder-Decoder simplificada.
    El codificador procesa la imagen y el prompt de entrada, y el decodificador genera la respuesta.
    """
    def __init__(self, vocab_size=2000, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Componentes de Entrada
        self.image_encoder = EncoderVisual(embed_dim)
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = CodificadorPosicional(embed_dim)
        
        # 2. Transformer Principal (Seq2Seq / Encoder-Decoder)
        # El Encoder procesa el contexto de entrada (Imagen + Prompt)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim*4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # El Decoder genera la respuesta autorregresiva usando cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim*4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 3. Capa de Salida a Vocabulario
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generar_mascara_causal(self, sz, device):
        """Genera una máscara triangular para evitar que el decoder mire hacia el futuro."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images, input_ids, target_ids, input_key_padding_mask=None, target_key_padding_mask=None):
        # 1. Obtener representación de la imagen y agregar dimensión de token -> (batch, 1, embed_dim)
        img_features = self.image_encoder(images).unsqueeze(1)
        
        # 2. Obtener embeddings del texto de entrada -> (batch, seq_len, embed_dim)
        text_features = self.token_embeddings(input_ids)
        text_features = self.pos_encoder(text_features)
        
        # 3. FUSIÓN: Concatenar imagen al principio del texto de entrada -> (batch, 1 + seq_len, embed_dim)
        # La imagen actúa como el primer "token visual"
        context_features = torch.cat([img_features, text_features], dim=1)
        
        # Procesar contexto por el Encoder del Transformer
        memory = self.transformer_encoder(context_features)
        
        # 4. Decodificar el Target (Respuesta)
        # Obtener embeddings del target
        target_features = self.token_embeddings(target_ids)
        target_features = self.pos_encoder(target_features)
        
        # Crear máscara causal para el autoregresor
        tgt_mask = self.generar_mascara_causal(target_ids.size(1), target_ids.device)
        
        # Decodificación
        output = self.transformer_decoder(
            tgt=target_features,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=target_key_padding_mask
        )
        
        # Proyectar al espacio del vocabulario
        return self.fc_out(output) # (batch, seq_len, vocab_size)

# Script de validación básico
if __name__ == "__main__":
    print("🤖 Instanciando modelo NetMindVLM...")
    modelo = NetMindVLM(vocab_size=8000, embed_dim=256)
    
    # Crear entradas de prueba (Batch size = 2, seq_len = 128)
    batch_size = 2
    seq_len = 128
    
    dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    dummy_inputs = torch.randint(0, 8000, (batch_size, seq_len))
    dummy_targets = torch.randint(0, 8000, (batch_size, seq_len))
    
    print("⏳ Realizando una pasada hacia adelante (forward pass) de prueba...")
    output = modelo(dummy_imgs, dummy_inputs, dummy_targets)
    
    print(f"✅ Pasada exitosa. Shape de salida del modelo: {output.shape}")
    # Debe coincidir con [batch_size, seq_len, vocab_size] -> [2, 128, 2000]
