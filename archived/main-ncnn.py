import time
import cv2
from ultralytics import YOLO
import ncnn

gpu_count = ncnn.get_gpu_count()
print(f"Vulkan GPU count detected by NCNN: {gpu_count}")

if gpu_count > 0:
    for i in range(gpu_count):
        device_info = ncnn.get_gpu_info(i)
        # In newer ncnn wrappers device_name is a property on gpu_info
        name = getattr(device_info, "device_name", f"GPU {i}")
        print(f" -> Device {i}: {name}")
else:
    print("WARNING: No Vulkan GPU detected! NCNN is running on CPU.")

class MovingAverage:
    """Exponential Moving Average (EMA) smoothed timer."""
    def __init__(self, alpha: float = 0.08):
        self.alpha = alpha
        self.value = None

    def update(self, val: float) -> float:
        if self.value is None:
            self.value = val
        else:
            self.value = self.alpha * val + (1.0 - self.alpha) * self.value
        return self.value


def main():
    # 1. Load the exported NCNN model directory
    model = YOLO("models/puck-eye-seg-s_ncnn_model")

    # 2. Open video source (0 for webcam or pass 'video.mp4')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Running YOLO Tracking via NCNN + Vulkan...")

    # Profilers (EMA smoothing)
    avg_cap = MovingAverage()
    avg_prep = MovingAverage()
    avg_infer = MovingAverage()
    avg_post = MovingAverage()
    avg_track = MovingAverage()
    avg_vis = MovingAverage()
    avg_total = MovingAverage()

    prev_frame_time = time.perf_counter()
    frame_count = 0

    while cap.isOpened():
        loop_start = time.perf_counter()

        # Measure capture time
        t_cap_start = time.perf_counter()
        success, frame = cap.read()
        t_cap = (time.perf_counter() - t_cap_start) * 1000.0

        if not success:
            break

        # Measure track() call
        t_track_start = time.perf_counter()
        results = model.track(
            source=frame,
            persist=True,
            device="vulkan:0",
            tracker="bytetrack.yaml",  # or "botsort.yaml"
            conf=0.35,
            iou=0.5,
            verbose=False
        )
        total_track_time = (time.perf_counter() - t_track_start) * 1000.0

        # Extract Ultralytics native timing breakdown (in ms)
        speed = getattr(results[0], "speed", {})
        prep_ms = speed.get("preprocess", 0.0)
        infer_ms = speed.get("inference", 0.0)
        post_ms = speed.get("postprocess", 0.0)
        # Any residual time in model.track() belongs to tracker association / wrapper overhead
        tracker_overhead_ms = max(0.0, total_track_time - (prep_ms + infer_ms + post_ms))

        # Measure visualization and display
        t_vis_start = time.perf_counter()
        annotated_frame = results[0].plot()

        # Update smooth timings
        c_ms = avg_cap.update(t_cap)
        p_ms = avg_prep.update(prep_ms)
        i_ms = avg_infer.update(infer_ms)
        po_ms = avg_post.update(post_ms)
        tr_ms = avg_track.update(tracker_overhead_ms)

        loop_end = time.perf_counter()
        t_loop = (loop_end - loop_start) * 1000.0
        tot_ms = avg_total.update(t_loop)
        fps = 1000.0 / tot_ms if tot_ms > 0 else 0.0

        # Draw metrics HUD on frame
        metrics = [
            f"FPS: {fps:5.1f} (Total: {tot_ms:5.1f}ms)",
            f"Cap/Read:     {c_ms:5.1f}ms",
            f"Preprocess:   {p_ms:5.1f}ms",
            f"Inference:    {i_ms:5.1f}ms",
            f"Postprocess:  {po_ms:5.1f}ms",
            f"Tracker OVH:  {tr_ms:5.1f}ms",
        ]
        
        y_offset = 25
        for line in metrics:
            cv2.putText(annotated_frame, line, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 22

        cv2.imshow("AMD Radeon 890M - NCNN Vulkan Tracking", annotated_frame)
        avg_vis.update((time.perf_counter() - t_vis_start) * 1000.0)

        # Log metrics to console periodically (every 30 frames)
        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"[FPS: {fps:4.1f}] | "
                f"Cap: {c_ms:4.1f}ms | "
                f"Prep: {p_ms:4.1f}ms | "
                f"Infer: {i_ms:4.1f}ms | "
                f"Post: {po_ms:4.1f}ms | "
                f"Tracker: {tr_ms:4.1f}ms | "
                f"Render/GUI: {avg_vis.value:4.1f}ms"
            )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()