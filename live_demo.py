import cv2
import argparse
import time

from ultralytics import YOLO
import supervision as sv

# Parse the control arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1920, 1080], 
        nargs=2, 
        type=int
    )
    parser.add_argument(
        "--webcam-refresh-rate",
        default=.2,
        type=float
    )
    args = parser.parse_args()
    return args

# Annotation style to use to label the detections
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

# Connect to the logitech webcam and set the resolution
def initialize_webcam(webcam_resolution) -> cv2.VideoCapture:
    frame_width, frame_height = webcam_resolution
    vid_cap = cv2.VideoCapture(0)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return vid_cap

def main():
    args = parse_arguments()
    refresh_rate = args.webcam_refresh_rate

    video_capture = initialize_webcam(args.webcam_resolution)

    model = YOLO("yolov8n.pt")

    while True:
        time.sleep(refresh_rate)
        ret, frame = video_capture.read()
        if not ret: break # end of video stream

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )  
        
        cv2.imshow("yolov8 COCO", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()