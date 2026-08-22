import cv2
from ultralytics import YOLO

def main():
    # 1. Load the exported NCNN model directory
    # Ultralytics natively handles NCNN inference and tracker association (ByteTrack/BoT-SORT)
    model = YOLO("models/puck-eye-seg-s_ncnn_model")

    # 2. Open video source (0 for webcam or pass 'video.mp4')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Running YOLO Tracking via NCNN + Vulkan...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. Perform object tracking
        # persist=True ensures track IDs are maintained across frames
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml", # or "botsort.yaml"
            conf=0.35,
            iou=0.5,
            verbose=False
        )

        # 4. Visualize the tracking boxes and IDs on the frame
        annotated_frame = results[0].plot()

        # Display output
        cv2.imshow("AMD Radeon 890M - NCNN Vulkan Tracking", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()