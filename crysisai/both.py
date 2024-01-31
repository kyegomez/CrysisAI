from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (
    int(cap.get(x))
    for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    )
)

# Video writer
video_writer = cv2.VideoWriter(
    "heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(
    colormap=cv2.COLORMAP_PARULA,
    imw=w,
    imh=h,
    view_img=True,
    shape="circle",
)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        tracks = model.track(frame, persist=True, show=False)

        # Generate heatmap
        frame = heatmap_obj.generate_heatmap(frame, tracks)

        # Visualize the results on the frame
        annotated_frame = tracks[0].plot()

        # Overlay the heatmap and the object detection
        overlay_frame = cv2.addWeighted(frame, 0.6, annotated_frame, 0.4, 0)

        # Write the frame with heatmap to the video file
        video_writer.write(overlay_frame)

        # Display the overlay frame
        cv2.imshow("YOLOv8 Tracking with Heatmap", overlay_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        print(
            "Video frame is empty or video processing has been"
            " successfully completed."
        )
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()