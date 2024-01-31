from collections import defaultdict
from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2
import numpy as np
from pytube import YouTube


class CrysisAI:
    def __init__(
        self, model_name: str = "yolov8n.pt", *args, **kwargs
    ):
        self.model_name = model_name

        self.model = YOLO(model_name, *args, **kwargs)

    def download_video(self, video):
        return YouTube(video).streams.first().download()

    def heatmaps(self, video: str):
        cap = cv2.VideoCapture(video)
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
            "heatmap_output.avi",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
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

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print(
                    "Video frame is empty or video processing has"
                    " been successfully completed."
                )
                break
            tracks = self.model.track(im0, persist=True, show=False)

            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    def object_detection(self, video: str):
        # video_path = self.download_video(video)
        video_path = video
        cap = cv2.VideoCapture(video_path)

        # Store the track history
        track_history = defaultdict(lambda: [])

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = self.model.track(frame, persist=True)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append(
                        (float(x), float(y))
                    )  # x, y center point
                    if (
                        len(track) > 30
                    ):  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = (
                        np.hstack(track)
                        .astype(np.int32)
                        .reshape((-1, 1, 2))
                    )
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=10,
                    )

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def __call__(self, video: str):
        # Download the video from the URL
        video = self.download_video(video)

        # Initialize the video capture
        cap = cv2.VideoCapture(video)
        assert cap.isOpened(), "Error reading video file"

        # Get video properties
        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )

        # Initialize the video writer
        video_writer = cv2.VideoWriter(
            "heatmap_output.avi",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        # Initialize the heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(
            colormap=cv2.COLORMAP_PARULA,
            imw=w,
            imh=h,
            view_img=True,
            shape="circle",
        )

        # Initialize the track history
        track_history = defaultdict(lambda: [])

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print(
                    "Video frame is empty or video processing has"
                    " been successfully completed."
                )
                break

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = self.model.track(im0, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append(
                    (float(x), float(y))
                )  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = (
                    np.hstack(track)
                    .astype(np.int32)
                    .reshape((-1, 1, 2))
                )
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=10,
                )

            # Define tracks
            tracks = [
                {"id": track_id, "box": box.tolist()}
                for track_id, box in zip(track_ids, boxes)
            ]

            # Generate heatmap
            heatmap_frame = heatmap_obj.generate_heatmap(im0, tracks)

            # Overlay object detection on heatmap
            overlay = cv2.addWeighted(
                annotated_frame, 0.6, heatmap_frame, 0.4, 0
            )

            video_writer.write(overlay)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        return (
            "Processing completedp Output video: heatmap_output.avi"
        )


model = CrysisAI()

out = model.heatmaps("video.mp4")
print(out)
