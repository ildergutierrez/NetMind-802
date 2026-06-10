# NetMind-802
NetMind-802 es un modelo de inteligencia artificial multimodal pequeño, especializado, y diseñado desde cero en Python. Su propósito principal es actuar como un asistente de diagnóstico y seguridad telemática para redes cableadas e inalámbricas bajo estándares IEEE 802.

http://127.0.0.1:8000/
│
├── GET  /                ← Bienvenida: muestra nombre, versión y lista de endpoints
├── GET  /status          ← Estado del servidor y si el modelo está entrenado
├── POST /diagnose        ← Diagnóstico de un problema de red (texto + imagen opcional)
└── POST /train           ← Re-entrenar el modelo cuando añadas nuevos casos al dataset

1. Verifica e instala dependencias (dependencias.py)
2. Arranca el servidor FastAPI en http://127.0.0.1:8000
3. Al iniciar, verifica si existe el modelo entrenado (models/netmind_vlm.pth)
   └── Si NO existe → lanza el entrenamiento automáticamente
   └── Si SÍ existe → queda listo para responder consultas