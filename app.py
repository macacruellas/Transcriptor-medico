"""
Servidor web para transcripción médica con IA.
Versión con transcripción en tiempo real via Realtime API de OpenAI.
"""

import os
import json
import tempfile
import asyncio
import base64
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'transcripcion-medica-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROTOCOLOS_FILE = "protocolos.json"
active_sessions = {}


# ============================================================
# PROTOCOLOS
# ============================================================

def cargar_protocolos():
    if os.path.exists(PROTOCOLOS_FILE):
        with open(PROTOCOLOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        protocolos = [
            {
                "id": 1,
                "nombre": "Informe Citopatológico - CIGOR",
                "plantilla": "CIGOR\nCentro Integral de Ginecología, Obstetricia y Reproducción\n\nInforme Citopatológico\n\nNº LABORATORIO: ___\nPERTENECIENTE A: ___\nEDAD: ___\nSOLICITADO POR: ___\nFECHA: ___\nESTUDIO: CITOLOGICO\nMATERIAL: ___\n\n--- Diagnóstico Oncológico ---\n\nCÉLULAS PAVIMENTOSAS:\n___\n\nOTRAS CÉLULAS:\n___\n\nASPECTO DEL FONDO DEL FROTIS:\n___\n\n--- Cepillado Endocervical ---\n___\n\nClase: ___\n\n--- Bacterioscopia de Exudado Vaginal ---\n___\n\n--- Valoración Funcional ---\n___\n\nNOTA: ___",
                "campos": ["Nº Laboratorio", "Perteneciente a", "Edad", "Solicitado por", "Fecha", "Material", "Células pavimentosas", "Otras células", "Aspecto del fondo del frotis", "Cepillado endocervical", "Clase", "Bacterioscopia de exudado vaginal", "Valoración funcional", "Nota"]
            }
        ]
        guardar_protocolos(protocolos)
        return protocolos


def guardar_protocolos(protocolos):
    with open(PROTOCOLOS_FILE, "w", encoding="utf-8") as f:
        json.dump(protocolos, f, ensure_ascii=False, indent=2)


# ============================================================
# RUTAS WEB
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/protocolos", methods=["GET"])
def obtener_protocolos():
    return jsonify(cargar_protocolos())

@app.route("/protocolos", methods=["POST"])
def crear_protocolo():
    data = request.get_json()
    protocolos = cargar_protocolos()
    nuevo_id = max([p["id"] for p in protocolos], default=0) + 1
    nuevo = {"id": nuevo_id, "nombre": data["nombre"], "plantilla": data["plantilla"], "campos": data.get("campos", [])}
    protocolos.append(nuevo)
    guardar_protocolos(protocolos)
    return jsonify(nuevo)

@app.route("/protocolos/<int:protocolo_id>", methods=["DELETE"])
def eliminar_protocolo(protocolo_id):
    protocolos = cargar_protocolos()
    protocolos = [p for p in protocolos if p["id"] != protocolo_id]
    guardar_protocolos(protocolos)
    return jsonify({"ok": True})

@app.route("/interpretar", methods=["POST"])
def interpretar():
    data = request.get_json()
    texto = data.get("texto", "")
    protocolo = data.get("protocolo", None)
    if not texto:
        return jsonify({"error": "No se recibió texto"}), 400

    prompt_sistema = "Sos un asistente especializado en informes médicos.\nTu tarea es recibir la transcripción cruda de un dictado médico y generar un informe limpio."

    if protocolo:
        prompt_sistema += f"\n\nPLANTILLA/PROTOCOLO A SEGUIR:\n{protocolo['plantilla']}\n\nCAMPOS QUE DEBÉS COMPLETAR:\n{', '.join(protocolo.get('campos', []))}\n\nINSTRUCCIONES ESPECÍFICAS:\n1. Usá EXACTAMENTE la estructura de la plantilla proporcionada.\n2. Completá los campos marcados con '___' usando la información del dictado.\n3. Si el médico no mencionó un campo, dejalo con '___' (NO inventés información).\n4. Respetá los títulos, subtítulos y secciones tal cual están en la plantilla.\n5. Si el médico da instrucciones de edición (como 'borrá eso', 'cambiá X por Y'), ejecutalas.\n6. Corregí errores menores de transcripción sin cambiar el sentido médico.\n7. Devolvé SOLO el informe completado, sin explicaciones adicionales."
    else:
        prompt_sistema += "\n\nINSTRUCCIONES:\n1. INTERPRETAR Y EJECUTAR las instrucciones de edición del dictado.\n2. ESTRUCTURAR el informe en secciones (HALLAZGOS, CONCLUSIÓN, RECOMENDACIONES).\n3. CORREGIR errores menores de transcripción.\n4. DEVOLVER SOLO el informe limpio.\nIMPORTANTE: No inventés información que el médico no haya dicho."

    try:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Transcripción cruda del dictado:\n\n{texto}"}
            ],
            temperature=0.2
        )
        return jsonify({"informe": respuesta.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# TRANSCRIPCIÓN EN TIEMPO REAL VIA WEBSOCKET
# ============================================================

def run_realtime_session(sid):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(realtime_session(sid))
    except Exception as e:
        socketio.emit("transcription_error", {"error": str(e)}, room=sid)
    finally:
        loop.close()


async def realtime_session(sid):
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        async with async_client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2024-10-01"
        ) as connection:

            await connection.session.update(session={
                "instructions": "Eres un transcriptor médico. Solo devuelve el texto de lo que escuchas.",
                "modalities": ["text"],
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                    "language": "es"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            })

            active_sessions[sid] = {
                "connection": connection,
                "running": True,
                "transcripcion": "",
                "audio_buffer": None
            }

            socketio.emit("transcription_ready", room=sid)

            async def send_audio():
                while active_sessions.get(sid, {}).get("running", False):
                    audio_data = active_sessions[sid].get("audio_buffer", None)
                    if audio_data:
                        active_sessions[sid]["audio_buffer"] = None
                        try:
                            await connection.input_audio_buffer.append(audio=audio_data)
                        except Exception:
                            break
                    await asyncio.sleep(0.01)

            async def receive_events():
                try:
                    async for event in connection:
                        if not active_sessions.get(sid, {}).get("running", False):
                            break
                        if event.type == "conversation.item.input_audio_transcription.delta":
                            delta = event.delta
                            active_sessions[sid]["transcripcion"] += delta
                            socketio.emit("transcription_delta", {"delta": delta}, room=sid)
                        elif event.type == "error":
                            socketio.emit("transcription_error", {"error": str(event.error)}, room=sid)
                except Exception as e:
                    if active_sessions.get(sid, {}).get("running", False):
                        socketio.emit("transcription_error", {"error": str(e)}, room=sid)

            await asyncio.gather(send_audio(), receive_events())

    except Exception as e:
        socketio.emit("transcription_error", {"error": str(e)}, room=sid)
    finally:
        if sid in active_sessions:
            del active_sessions[sid]


@socketio.on("connect")
def handle_connect():
    print(f"  Cliente conectado: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in active_sessions:
        active_sessions[sid]["running"] = False
    print(f"  Cliente desconectado: {sid}")

@socketio.on("start_transcription")
def handle_start_transcription():
    sid = request.sid
    print(f"  Iniciando transcripción en vivo para: {sid}")
    thread = threading.Thread(target=run_realtime_session, args=(sid,), daemon=True)
    thread.start()

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    sid = request.sid
    if sid in active_sessions:
        active_sessions[sid]["audio_buffer"] = data["audio"]

@socketio.on("stop_transcription")
def handle_stop_transcription():
    sid = request.sid
    transcripcion = ""
    if sid in active_sessions:
        transcripcion = active_sessions[sid].get("transcripcion", "")
        active_sessions[sid]["running"] = False
    socketio.emit("transcription_stopped", {"full_text": transcripcion}, room=sid)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SERVIDOR DE TRANSCRIPCIÓN MÉDICA")
    print("  Transcripción en TIEMPO REAL")
    print("=" * 60)
    print()
    print("  Abrí tu navegador en: http://localhost:5000")
    print("  Para cerrar el servidor: Ctrl+C")
    print()
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
# prueba test issue #1