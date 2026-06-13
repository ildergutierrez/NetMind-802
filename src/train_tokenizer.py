"""
train_tokenizer.py — Entrena un tokenizer WordPiece de calidad
para texto técnico en español (redes, Cisco IOS, Linux, Windows).

Ejecutar UNA VEZ antes de reentrenar el modelo:
    python train_tokenizer.py

Genera: src/tokenizer_redes.json
"""
import os
import json
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import BertNormalizer

# ── Rutas ────────────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
dataset_path = os.path.join(project_root, "data", "dataset_redes.json")
out_path     = os.path.join(current_dir, "tokenizer_redes.json")

VOCAB_SIZE = 8000   # Suficiente para español técnico sin fragmentar palabras

# ── Cargar corpus desde el dataset ───────────────────────────────────────────
print("📖 Cargando dataset...")
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

textos = []
for caso in data:
    if caso.get("prompt"):  textos.append(caso["prompt"])
    if caso.get("target"):  textos.append(caso["target"])
    if caso.get("titulo"):  textos.append(caso["titulo"])

print(f"   {len(textos)} textos cargados.")

# ── Texto técnico adicional para enriquecer el vocabulario ───────────────────
extra = [
    # Términos de red más comunes
    "GigabitEthernet FastEthernet Serial Loopback Vlan BVI Tunnel",
    "switchport trunk access vlan native allowed mode",
    "spanning-tree portfast bpduguard bpdufilter rootguard",
    "ip route dhcp snooping arp inspection nat access-list",
    "router ospf eigrp bgp rip isis mpls",
    "show running-config startup-config interfaces status",
    "configure terminal no shutdown ip address subnet mask",
    "port-security violation shutdown restrict protect sticky",
    "DIAGNÓSTICO CAUSAS SOLUCIÓN VERIFICACIÓN RESULTADO",
    "error-disabled err-disabled reconvergencia adyacencia",
    "IEEE 802.1Q 802.1X 802.11 802.3 dot1x dot3svc",
    "VLAN trunk encapsulation mismatch nativa",
    "DHCP pool scope lease binding conflict excluded-address",
    "ARP spoofing poisoning flooding broadcast storm loop",
    "firewall ACL permit deny access-group access-list",
    "VPN IPsec IKE ISAKMP phase tunnel crypto map transform-set",
    "OSPF neighbor hello dead area backbone LSA SPF",
    "BGP neighbor AS-path prefix-list route-map community",
    "WiFi SSID BSSID canal interference 2.4GHz 5GHz WPA2 WPA3",
    "NAT overload inside outside translation pool",
    "QoS DSCP priority queue class-map policy-map service-policy",
    "STP RSTP rapid-pvst priority root bridge port role state",
    "EtherChannel LACP PAgP port-channel bundle aggregate",
    "interface GigabitEthernet0/0 GigabitEthernet1/0/1",
    "ip dhcp snooping trust ip arp inspection vlan",
    "show ip route ospf eigrp bgp connected static",
    "debug ip ospf adj events debug spanning-tree events",
    "configure terminal interface vlan shutdown no shutdown",
    "Windows PowerShell netsh wlan show interfaces profiles",
    "Linux bash ethtool nmcli brctl iptables ip neighbor",
    "Get-NetAdapter Set-NetAdapterAdvancedProperty Start-Service",
    "sudo systemctl restart ip neigh del arp-scan",
]
textos.extend(extra)

# ── Construir y entrenar el tokenizer ────────────────────────────────────────
print(f"🔧 Entrenando tokenizer WordPiece (vocab={VOCAB_SIZE})...")

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer    = BertNormalizer(lowercase=False)  # preservar mayúsculas (Cisco IOS)
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=1,
    continuing_subword_prefix="##",
)

tokenizer.train_from_iterator(textos, trainer=trainer)
tokenizer.save(out_path)

# ── Verificación ─────────────────────────────────────────────────────────────
print(f"\n✅ Tokenizer guardado en: {out_path}")
print(f"   Vocab size real: {tokenizer.get_vocab_size()}")

pruebas = [
    "GigabitEthernet port-security violation",
    "El puerto está en err-disabled por violación de port-security",
    "show ip ospf neighbor",
    "DIAGNÓSTICO: Puerto en estado err-disable",
    "configure terminal interface GigabitEthernet1/0/5 shutdown no shutdown",
]
print("\n🔍 Prueba de tokenización:")
for txt in pruebas:
    tokens = tokenizer.encode(txt).tokens
    print(f"   '{txt[:50]}...' → {tokens[:12]}{'...' if len(tokens)>12 else ''}")