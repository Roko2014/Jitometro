import ultralytics
import cv2
from ultralytics.utils.downloads import safe_download
from ultralytics import solutions

ultralytics.checks()

cap = cv2.VideoCapture("Macizo 1 1Corte.mp4") # Replace with your video file path
assert cap.isOpened(), "Error reading video file"

# Define region points
region_points = [(850, 0), (850, 1080)] # For vertical line counting
# region_points = [(20, 400), (1080, 400)]  # For horizontal line counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # For rectangle region counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

#Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("Macizo_1_1Corte_Conteo_14Julio2026.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))
# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="best_jitomate_v1_17032026.pt",  # model="yolo26n-obb.pt" for object counting using YOLO26 OBB model.
    classes=[3, 4, 5, 6],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    show_in=True,  # Display in counts
    # show_out=True,  # Display out counts
    # line_width=2,  # Adjust the line width for bounding boxes and text display
)
# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = counter(im0)  # count the objects
    video_writer.write(results.plot_im)   # write the video frames

cap.release()   # Release the capture
video_writer.release()