"""
=======================================================
 web_search.py — Módulo de búsqueda web (Doble motor)
 Motor 1: Wikipedia REST API — Definiciones técnicas de redes
 Motor 2: DuckDuckGo HTML — Resultados web adicionales
 Ninguno requiere API Key ni autenticación.
=======================================================
"""

import sys
import httpx
import urllib.parse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# APIs públicas sin autenticación
WIKIPEDIA_API  = "https://en.wikipedia.org/api/rest_v1/page/summary/"
DDG_HTML       = "https://html.duckduckgo.com/html/"

HEADERS = {
    "User-Agent": "NetMind-802/1.0 (Network AI Diagnostic Tool; educational use)"
}


def buscar_wikipedia(query: str) -> dict:
    """
    Busca un resumen técnico en Wikipedia (REST API pública).
    Ideal para términos de redes como: VLAN, STP, DHCP, IEEE 802.1X, etc.
    """
    try:
        # Normalizar la consulta para la URL de Wikipedia
        termino = urllib.parse.quote(query.replace(" ", "_"))
        url = f"{WIKIPEDIA_API}{termino}"

        with httpx.Client(timeout=8.0) as client:
            r = client.get(url, headers=HEADERS)

        if r.status_code == 200:
            data = r.json()
            resumen = data.get("extract", "").strip()
            fuente  = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            if resumen:
                return {
                    "encontrado" : True,
                    "resumen"    : resumen[:600],  # Limitar a 600 caracteres
                    "fuente_url" : fuente,
                    "resultados" : []
                }

        return {"encontrado": False, "resumen": "", "fuente_url": "", "resultados": []}

    except Exception:
        return {"encontrado": False, "resumen": "", "fuente_url": "", "resultados": []}


def buscar_duckduckgo_html(query: str, max_resultados: int = 3) -> list:
    """
    Extrae resultados de búsqueda de DuckDuckGo vía la interfaz HTML.
    Retorna una lista de dicts {titulo, url, snippet}.
    """
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            r = client.post(
                DDG_HTML,
                data={"q": query, "b": "", "kl": "es-es"},
                headers=HEADERS
            )

        if r.status_code != 200:
            return []

        # Parseo manual liviano: buscar bloques de resultado
        html  = r.text
        items = []
        # DuckDuckGo HTML devuelve resultados en divs con class="result__body"
        # Extraemos entre <a class="result__a" href="...">
        import re
        # Extraer URLs y snippets de forma básica
        urls = re.findall(r'class="result__a"[^>]*href="(https?://[^"]+)"', html)
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)<', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', html)

        for i in range(min(len(urls), max_resultados)):
            items.append({
                "titulo"  : titles[i].strip()  if i < len(titles)   else "Sin título",
                "url"     : urls[i].strip(),
                "snippet" : snippets[i].strip() if i < len(snippets) else ""
            })

        return items

    except Exception:
        return []


def buscar_en_web(query: str, max_resultados: int = 3) -> dict:
    """
    Motor de búsqueda doble:
      1. Intenta Wikipedia para obtener una definición técnica estructurada.
      2. Complementa con resultados de DuckDuckGo HTML.

    Args:
        query:          Consulta o descripción del problema de red.
        max_resultados: Máximo de resultados de DuckDuckGo a incluir.

    Returns:
        dict con encontrado, resumen, fuente_url, resultados.
    """
    # Motor 1: Wikipedia
    wiki = buscar_wikipedia(query)

    # Motor 2: DuckDuckGo HTML (siempre como complemento)
    ddg_items = buscar_duckduckgo_html(query, max_resultados)

    encontrado = wiki["encontrado"] or len(ddg_items) > 0

    return {
        "encontrado" : encontrado,
        "resumen"    : wiki["resumen"] if wiki["encontrado"] else "Sin resumen directo disponible.",
        "fuente_url" : wiki["fuente_url"],
        "resultados" : ddg_items
    }


def construir_contexto_web(query: str) -> str:
    """
    Retorna un bloque de texto formateado con información de la web
    para enriquecer la respuesta del modelo de IA.
    """
    resultado = buscar_en_web(query)

    if not resultado["encontrado"]:
        return ""

    lineas = ["\n[CONTEXTO WEB ACTUALIZADO]"]

    if resultado["resumen"] and resultado["resumen"] != "Sin resumen directo disponible.":
        lineas.append(f"Definicion tecnica: {resultado['resumen']}")

    if resultado["fuente_url"]:
        lineas.append(f"Fuente: {resultado['fuente_url']}")

    if resultado["resultados"]:
        lineas.append("Resultados relacionados:")
        for r in resultado["resultados"]:
            lineas.append(f"  - {r['titulo']}: {r['url']}")
            if r.get("snippet"):
                lineas.append(f"    {r['snippet'][:120]}")

    return "\n".join(lineas)


# Prueba directa del módulo
if __name__ == "__main__":
    tests = [
        "IEEE 802.1X port-based authentication",
        "VLAN trunking 802.1Q",
        "Spanning Tree Protocol STP loop"
    ]
    for consulta in tests:
        print(f"\nBuscando: '{consulta}'")
        print("-" * 50)
        ctx = construir_contexto_web(consulta)
        print(ctx if ctx else "Sin resultados.")
