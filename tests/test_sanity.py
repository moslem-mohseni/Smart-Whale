import asyncio
import sys
import os

from ai.data.collectors.base.collector import BaseCollector
from ai.data.collectors.text.specialized.wiki_collector import WikiCollector


async def test_wiki_collector():
    """Prueba del recolector de Wikipedia utilizando la clase WikiCollector"""
    # Obtener el título del argumento de la línea de comandos o usar el valor predeterminado
    title = sys.argv[1] if len(sys.argv) > 1 else "هوش مصنوعی"

    # Crear una instancia de la clase WikiCollector
    collector = WikiCollector(language="fa", max_length=3000)
    collector.set_title(title)

    print(f"🔍 Buscando artículo '{title}' en Wikipedia...")
    result = await collector.collect_data()

    if result:
        print("\n✅ Resultado de la recopilación:")
        print(f"Título: {result['title']}")
        print(f"Idioma: {result['language']}")
        print(f"Longitud del contenido: {result['length']} caracteres")
        print(f"Hora: {result['timestamp']}")
        print("\nFragmento del contenido:")
        print(result['content'][:500] + "...\n")
        return True
    else:
        print("❌ La recopilación de información falló.")
        return False


# Ejecutar la prueba
if __name__ == "__main__":
    success = asyncio.run(test_wiki_collector())
    if success:
        print("✅ La prueba se completó con éxito.")
    else:
        print("❌ La prueba falló.")
        sys.exit(1)