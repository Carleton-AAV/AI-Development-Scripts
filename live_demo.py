import cv2
import argparse
import time
import numpy as np

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
        "--update_rate",
        default=.05,
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

class DemoModel:
    def __init__(self, weights_file: str, demo_label: str):
        self.model = YOLO(weights_file)
        self.class_labels = self.model.model.names
        self.demo_label =  demo_label

    # Run the model on the frame and return the detections
    def inferrence(self, frame: np.ndarray) -> sv.Detections:
        result = self.model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        return detections

    # Match detected objects to their labels in the model
    def generate_labels(self, detections: sv.Detections) -> list:
        labels = [
            f"{self.class_labels[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        return labels
    
    @staticmethod
    def annotate_frame(frame: np.ndarray, detections: sv.Detections, labels: list) -> np.ndarray:
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )  
        return frame

    def detect_and_show(self, frame: np.ndarray) -> None:
        detections = self.inferrence(frame)
        labels = self.generate_labels(detections)
        annotated_frame = self.annotate_frame(frame, detections, labels)
        cv2.imshow(self.demo_label, annotated_frame)

models = [ 
    DemoModel("yolov8n.pt", "Benchmark"),
    DemoModel("test_trained.pt", "Proof of Concept"),
]


def main():
    args = parse_arguments()
    video_capture = initialize_webcam(args.webcam_resolution)


    while True:
        # The update wait is within this loops such that compute budget isn't exceeded 
        # with multiple models and that it only scales with the update rate
        for model in models:
            time.sleep(args.update_rate)
            ret, frame = video_capture.read()

            if not ret: break # end of video stream
            model.detect_and_show(frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()