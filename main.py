import cv2
import pandas as pd
from ultralytics import YOLO, solutions

# 1. Cargar el modelo
model = YOLO("best_jitomate_v1_17032026.pt")

# 2. Configurar captura de video
cap = cv2.VideoCapture("Macizo_1_1Corte.MOV")
assert cap.isOpened(), "Error al abrir el video"

w, h, fps = (int(cap.get(x)) for x in [cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS])

# 3. Definir línea de conteo VERTICAL a la derecha
# Multiplicamos por 0.85 para situarla donde dibujaste tu línea azul
line_x = int(w * 0.85)
line_points = [(line_x, 0), (line_x, h)]

# 4. Inicializar el Contador de Ultralytics
counter = solutions.ObjectCounter(
    show=True,                # Muestra el video en tiempo real
    region=line_points,          # Línea vertical a la derecha
    names=model.names,            # ['3', '4', '5', '6']
    draw_tracks=True,             # Dibuja el rastro del seguimiento
    line_thickness=2,
)

print(f"Procesando con línea de conteo en X={line_x} (Zona de mayor claridad)...")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # Tracking con persistencia
    # classes=[0, 1, 2, 3] asegura que solo rastree tus 4 categorías
    tracks = model.track(im0, persist=True, show=False, classes=[0, 1, 2, 3])

    # Aplicar lógica de conteo
    im0 = counter.start_counting(im0, tracks)

    # Presiona 'q' para cerrar la ventana manualmente si es necesario
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# 5. Exportación de resultados a Excel
reporte_datos = []
for label, counts in counter.class_wise_count.items():
    # Sumamos IN y OUT para obtener el total de cruces en esa línea
    total_clase = counts.get('IN', 0) + counts.get('OUT', 0)
    reporte_datos.append({"Nivel de Madurez": label, "Total": total_clase})

df = pd.DataFrame(reporte_datos)
df.to_excel("Reporte_Jitomates_Primer_Plano.xlsx", index=False)

print("\n" + "="*40)
print(f"Reporte generado: Reporte_Jitomates_Primer_Plano.xlsx")
print("="*40)