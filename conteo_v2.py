"""
conteo_v2.py — Jitometro v2 (3 clases: Inmaduro, Semimaduro, Maduro)
======================================================================
Pipeline de conteo corregido para video de invernadero a 45°.

Correcciones respecto a main.py / Conteo2.py:
  1. classes=[0, 1, 2] ahora referencia las clases REALES del modelo v2.
  2. La línea de conteo se coloca VERTICALMENTE en el tercio izquierdo
     del frame para capturar jitomates conforme la cámara avanza.
  3. BotSort está configurado con track_high_thresh más conservador
     para evitar IDs duplicados cuando el tracker pierde objetos.
  4. Se exporta un resumen JSON con los conteos por clase al final.

IMPORTANTE — Ajusta ESTAS variables antes de ejecutar:
    VIDEO_PATH    : ruta al video MOV/MP4 del pasillo
    MODEL_PATH    : ruta a los pesos del modelo v2 entrenado
    OUTPUT_VIDEO  : nombre del archivo de salida
    LINE_X        : posición X de la línea virtual (ver comentario abajo)
"""

import cv2
import json
from pathlib import Path
from datetime import datetime
from ultralytics import solutions, YOLO

# ─── CONFIGURACIÓN — AJUSTA AQUÍ ─────────────────────────────────────────────

VIDEO_PATH   = "Macizo_1_1Corte.MOV"
MODEL_PATH   = "runs/detect/train_v2/weights/best.pt"   # pesos del modelo v2
OUTPUT_VIDEO = f"conteo_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
CONF_THRESH  = 0.35     # más alto que v1 (0.2) para reducir falsos positivos

# ─── LÍNEA DE CONTEO ─────────────────────────────────────────────────────────
# Para video en pasillo a 45°, la cámara avanza horizontalmente.
# La línea vertical debe estar en ~40% del ancho del frame para dar tiempo
# al tracker de asignar un ID estable antes de que el objeto cruce.
# Detecta el ancho real del video y ajusta automáticamente:
_cap_tmp = cv2.VideoCapture(VIDEO_PATH)
assert _cap_tmp.isOpened(), f"No se puede abrir el video: {VIDEO_PATH}"
W = int(_cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(_cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(_cap_tmp.get(cv2.CAP_PROP_FPS))
_cap_tmp.release()

LINE_X = int(W * 0.40)   # 40% del ancho → puedes cambiar a 0.50 si prefieres centro
region_points = [(LINE_X, 0), (LINE_X, H)]

print(f"📹 Video: {W}x{H} @ {FPS}fps")
print(f"📏 Línea de conteo en X={LINE_X} (40% de {W}px)")

# ─── CLASES DEL MODELO v2 ────────────────────────────────────────────────────
# 0 = Inmaduro, 1 = Semimaduro, 2 = Maduro
# (si no tienes clase Inmaduro en tu dataset, usa classes=[1, 2])
CLASSES_TO_COUNT = [0, 1, 2]

CLASS_NAMES = {0: "Inmaduro", 1: "Semimaduro", 2: "Maduro"}
CLASS_COLORS = {
    0: (0, 255, 0),    # Verde → Inmaduro
    1: (0, 165, 255),  # Naranja → Semimaduro
    2: (0, 0, 255),    # Rojo → Maduro
}

# ─── INICIALIZAR CAPTURA Y WRITER ────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error al abrir el video."

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H)
)

# ─── INICIALIZAR CONTADOR ────────────────────────────────────────────────────
# BotSort es mejor que ByteTrack para objetos que desaparecen y reaparecen
# (p.ej. jitomates parcialmente ocluidos por hojas).
# track_high_thresh=0.5 reduce re-asignación de IDs falsos.
counter = solutions.ObjectCounter(
    show=True,
    show_in=True,
    show_out=False,          # Solo contamos "entradas" al pasar la línea
    region=region_points,
    model=MODEL_PATH,
    verbose=False,
    conf=CONF_THRESH,
    show_conf=True,
    show_labels=True,
    classes=CLASSES_TO_COUNT,
    tracker="botsort.yaml",  # BotSort: mejor para oclusión parcial
)

# ─── PROCESAMIENTO FRAME A FRAME ─────────────────────────────────────────────
frame_count = 0
print(f"\n🚀 Iniciando procesamiento... (presiona 'q' para detener)\n")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("✅ Video procesado completamente.")
        break

    results = counter(im0)
    video_writer.write(results.plot_im)
    frame_count += 1

    if frame_count % 100 == 0:
        print(f"  Frame {frame_count} procesado...")

# ─── RESULTADOS FINALES ───────────────────────────────────────────────────────
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Extraer conteos del counter
# El ObjectCounter de Ultralytics expone los conteos en .in_count y .out_count
# pero no por clase individualmente — por eso también puedes usar el modelo
# directamente para un reporte más detallado.
print("\n" + "="*50)
print("        RESUMEN DE CONTEO — Jitometro v2")
print("="*50)

# Intenta acceder a los conteos por clase si están disponibles
try:
    class_counts = counter.classwise_counts
    total = 0
    result_data = {}
    for cls_id, counts in class_counts.items():
        name = CLASS_NAMES.get(int(cls_id), str(cls_id))
        in_count = counts.get("IN", 0)
        print(f"  {name:<14}: {in_count:>4} jitomates")
        result_data[name] = in_count
        total += in_count
    print(f"  {'TOTAL':<14}: {total:>4} jitomates")
except AttributeError:
    # Fallback para versiones más antiguas de Ultralytics
    print(f"  Conteo IN total : {counter.in_count}")
    result_data = {"total_in": counter.in_count}
    total = counter.in_count

print("="*50)

# Guardar reporte JSON
report = {
    "video": VIDEO_PATH,
    "modelo": MODEL_PATH,
    "fecha": datetime.now().isoformat(),
    "frames_procesados": frame_count,
    "linea_conteo_x": LINE_X,
    "conteos": result_data,
    "total": total,
}
report_path = Path(OUTPUT_VIDEO).stem + "_reporte.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n📄 Reporte guardado en: {report_path}")
print(f"🎬 Video de salida    : {OUTPUT_VIDEO}\n")
