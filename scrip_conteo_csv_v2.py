import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO, solutions

# ── Configuración ─────────────────────────────────────────────────────────────
VIDEO_PATH = "Macizo_Moctezuma/Macizo_1_2Corte.MOV"

model_check = YOLO("best_v2.pt")
CLASS_NAMES = model_check.names  # {0: 'Inmaduro', 1: 'Semimaduro', 2: 'Maduro'}
print("Clases del modelo:", CLASS_NAMES)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

region_points = [(1800, 3800), (1800, 0)]

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                         cv2.CAP_PROP_FRAME_HEIGHT,
                                         cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("Macizo_1_2Corte_Jitometro_v2.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    show=True,
    show_in=True,
    show_out=False,
    region=region_points,
    model="best_v2.pt",
    verbose=True,
    conf=0.2,
    show_conf=True,
    classes=[0, 1, 2],
    tracker="botsort.yaml",
    show_labels=True,
)

# ── Procesar video ────────────────────────────────────────────────────────────
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video procesado completamente.")
        break
    results = counter(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# ── DEBUG: ver estructura exacta del contador ─────────────────────────────────
print("classwise_count:", counter.classwise_count)  # sin 's'
print("in_count:",        counter.in_count)
print("out_count:",       counter.out_count)

# ── Exportar resultados ───────────────────────────────────────────────────────
rows = []
for idx in sorted(CLASS_NAMES.keys()):
    nombre = CLASS_NAMES[idx]
    data   = counter.classwise_count.get(nombre, {})
    conteo = data.get("IN", 0) if isinstance(data, dict) else data
    rows.append({
        "Clase":            nombre,
        "Índice":           idx,
        "Conteo":           conteo,
        "Archivo de video": VIDEO_PATH,
    })

df = pd.DataFrame(rows, columns=["Clase", "Índice", "Conteo", "Archivo de video"])
print(df)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = f"conteo_jitomates_{timestamp_str}.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado: {csv_path}")

xlsx_path = f"conteo_jitomates_{timestamp_str}.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Conteo")
print(f"Excel guardado: {xlsx_path}")