import cv2
import pandas as pd
from datetime import datetime
from ultralytics import solutions

# ── Configuración ─────────────────────────────────────────────────────────────
VIDEO_PATH = "Macizo_1_1Corte.MOV"

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

region_points = [(1800, 3800), (1800, 0)]

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                         cv2.CAP_PROP_FRAME_HEIGHT,
                                         cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("Pruebita19.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    show=True,
    show_in=True,
    show_out=False,
    region=region_points,
    model="runs/detect/train/weights/best.pt",
    verbose=True,
    conf=0.2,
    show_conf=True,
    tracker="botsort.yaml",
    show_labels=True,
)

# ── Procesar video (sin tocar nada) ──────────────────────────────────────────
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

print("classwise_counts:", counter.classwise_count)

# ── Exportar resultados al terminar ──────────────────────────────────────────
# Las clases de madurez que te interesan
CLASES_MADUREZ = {"3", "4", "5", "6"}

# counter.classwise_counts → {"3": {"IN": n, "OUT": n}, "4": {...}, ...}
# Solo tomamos las clases 3-6; si una no apareció en el video, conteo = 0
rows = []
for clase in sorted(CLASES_MADUREZ):
    conteo = 0
    if hasattr(counter, "classwise_count") and clase in counter.classwise_count:
        conteo = counter.classwise_count[clase].get("IN", 0)
    rows.append({
        "Clase":  clase,
        "Conteo": conteo,
        "Archivo de video": VIDEO_PATH,
    })

df = pd.DataFrame(rows, columns=["Clase", "Conteo", "Archivo de video"])

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV
csv_path = f"conteo_jitomates_{timestamp_str}.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV guardado: {csv_path}")

# Excel
xlsx_path = f"conteo_jitomates_{timestamp_str}.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Conteo")
print(f"Excel guardado: {xlsx_path}")