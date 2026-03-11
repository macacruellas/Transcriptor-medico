"""
Servidor web para transcripción médica con IA.
Versión con soporte de protocolos/plantillas.
"""

import os
import json
import tempfile
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Archivo donde se guardan los protocolos
PROTOCOLOS_FILE = "protocolos.json"


def cargar_protocolos():
    """Carga los protocolos guardados desde el archivo JSON."""
    if os.path.exists(PROTOCOLOS_FILE):
        with open(PROTOCOLOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Protocolo de ejemplo precargado
        protocolos = [
            {
                "id": 1,
                "nombre": "Informe Citopatológico - CIGOR",
                "plantilla": """CIGOR
Centro Integral de Ginecología, Obstetricia y Reproducción

Informe Citopatológico

Dra. Patricia M. Cafferata - M.P.11213 - Citopatóloga - C.E.3071
Dra. Ximena Martínez García - M.P. 21516 - Citología Exfoliativa - C.E.7878

Nº LABORATORIO: ___
PERTENECIENTE A: ___
EDAD: ___
SOLICITADO POR: ___
FECHA: ___
ESTUDIO: CITOLOGICO
MATERIAL: ___

--- Diagnóstico Oncológico ---

CÉLULAS PAVIMENTOSAS:
___

OTRAS CÉLULAS:
___

ASPECTO DEL FONDO DEL FROTIS:
___

--- Cepillado Endocervical ---
___

Clase: ___

--- Bacterioscopia de Exudado Vaginal ---
___

--- Valoración Funcional ---
___

NOTA: ___""",
                "campos": [
                    "Nº Laboratorio",
                    "Perteneciente a (nombre paciente)",
                    "Edad",
                    "Solicitado por",
                    "Fecha",
                    "Material",
                    "Células pavimentosas",
                    "Otras células",
                    "Aspecto del fondo del frotis",
                    "Cepillado endocervical",
                    "Clase",
                    "Bacterioscopia de exudado vaginal",
                    "Valoración funcional",
                    "Nota"
                ]
            }
        ]
        guardar_protocolos(protocolos)
        return protocolos


def guardar_protocolos(protocolos):
    """Guarda los protocolos en el archivo JSON."""
    with open(PROTOCOLOS_FILE, "w", encoding="utf-8") as f:
        json.dump(protocolos, f, ensure_ascii=False, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/protocolos", methods=["GET"])
def obtener_protocolos():
    """Devuelve la lista de protocolos guardados."""
    protocolos = cargar_protocolos()
    return jsonify(protocolos)


@app.route("/protocolos", methods=["POST"])
def crear_protocolo():
    """Crea un nuevo protocolo."""
    data = request.get_json()
    protocolos = cargar_protocolos()

    nuevo_id = max([p["id"] for p in protocolos], default=0) + 1
    nuevo = {
        "id": nuevo_id,
        "nombre": data["nombre"],
        "plantilla": data["plantilla"],
        "campos": data.get("campos", [])
    }

    protocolos.append(nuevo)
    guardar_protocolos(protocolos)
    return jsonify(nuevo)


@app.route("/protocolos/<int:protocolo_id>", methods=["DELETE"])
def eliminar_protocolo(protocolo_id):
    """Elimina un protocolo."""
    protocolos = cargar_protocolos()
    protocolos = [p for p in protocolos if p["id"] != protocolo_id]
    guardar_protocolos(protocolos)
    return jsonify({"ok": True})


@app.route("/transcribir", methods=["POST"])
def transcribir():
    """Recibe audio y lo transcribe con Whisper."""
    if "audio" not in request.files:
        return jsonify({"error": "No se recibió audio"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resultado = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="es"
            )
        return jsonify({"transcripcion": resultado.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_path)


@app.route("/interpretar", methods=["POST"])
def interpretar():
    """Interpreta la transcripción y genera el informe según el protocolo."""
    data = request.get_json()
    texto = data.get("texto", "")
    protocolo = data.get("protocolo", None)

    if not texto:
        return jsonify({"error": "No se recibió texto"}), 400

    # Prompt base
    prompt_sistema = """Sos un asistente especializado en informes médicos.
Tu tarea es recibir la transcripción cruda de un dictado médico y generar un informe limpio."""

    if protocolo:
        # Si hay protocolo, usarlo como plantilla
        prompt_sistema += f"""

PLANTILLA/PROTOCOLO A SEGUIR:
{protocolo['plantilla']}

CAMPOS QUE DEBÉS COMPLETAR:
{', '.join(protocolo.get('campos', []))}

INSTRUCCIONES ESPECÍFICAS:
1. Usá EXACTAMENTE la estructura de la plantilla proporcionada.
2. Completá los campos marcados con "___" usando la información del dictado.
3. Si el médico no mencionó un campo, dejalo con "___" (NO inventés información).
4. Respetá los títulos, subtítulos y secciones tal cual están en la plantilla.
5. Si el médico da instrucciones de edición (como "borrá eso", "cambiá X por Y"), ejecutalas.
6. Corregí errores menores de transcripción sin cambiar el sentido médico.
7. Devolvé SOLO el informe completado, sin explicaciones adicionales."""

    else:
        # Sin protocolo, comportamiento libre
        prompt_sistema += """

INSTRUCCIONES:
1. INTERPRETAR Y EJECUTAR las instrucciones de edición del dictado:
   - "Borrá lo anterior" / "Borrá la última oración" → eliminar lo indicado
   - "Cambiá [X] por [Y]" → hacer el reemplazo
   - "Corregí eso" / "No, quise decir..." → corregir según contexto
2. ESTRUCTURAR el informe en secciones (HALLAZGOS, CONCLUSIÓN, RECOMENDACIONES).
3. CORREGIR errores menores de transcripción.
4. DEVOLVER SOLO el informe limpio.

IMPORTANTE: No inventés información que el médico no haya dicho."""

    try:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Transcripción cruda del dictado:\n\n{texto}"}
            ],
            temperature=0.2
        )

        informe = respuesta.choices[0].message.content
        return jsonify({"informe": informe})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SERVIDOR DE TRANSCRIPCIÓN MÉDICA")
    print("  con soporte de protocolos")
    print("=" * 60)
    print()
    print("  Abrí tu navegador en: http://localhost:5000")
    print("  Para cerrar el servidor: Ctrl+C")
    print()
    app.run(debug=True, port=5000)
