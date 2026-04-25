import cv2
from torch.xpu import device

from ultralytics import solutions

cap = cv2.VideoCapture("Macizo_1_1Corte.MOV")
assert cap.isOpened(), "Error reading video file"

region_points = [(2000, 0), (2000, 1880)]                                      # line counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("Pruebita7.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,
    show_in=True,  # display the output
    show_out=False,
    region=region_points,  # pass region points
    model="runs/detect/train/weights/best.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    verbose=True,
    conf=0.2,
    show_conf=True,
    #classes=[1],  # count specific classes i.e. person and car with COCO pretrained model.
    tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
    show_labels=True
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output


    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows