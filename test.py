import time
import cv2
import numpy as np
import onnxruntime as ort
from main import (
    INPUT_SIZE,
    MODEL_PATH,
    coreml_options,
    letterbox,
    postprocess,
    session as migraphx_session,
)


def get_test_session():
    available_providers = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available_providers:
        test_session = ort.InferenceSession(
            MODEL_PATH,
            providers=[("CoreMLExecutionProvider", coreml_options), "CPUExecutionProvider"],
        )
        print("Using CoreMLExecutionProvider for the benchmark")
    else:
        test_session = migraphx_session
        print("CoreMLExecutionProvider unavailable; using the main session")

    return test_session, test_session.get_inputs()[0].name


def main():
    session, input_name = get_test_session()
    image_path = "sample.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    orig_shape = frame.shape[:2]
    num_runs = 100
    warmup_runs = 10

    print(f"Loaded '{image_path}' (shape: {frame.shape})")
    print(f"Performing {warmup_runs} warm-up runs...")

    # Warm-up runs
    for _ in range(warmup_runs):
        img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
        )
        outputs = session.run(None, {input_name: blob})
        _ = postprocess(outputs[0], orig_shape, ratio, pad)

    print(f"Benchmarking {num_runs} iterations (no tracker)...")

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    final_boxes = None
    final_confs = None
    final_class_ids = None

    for _ in range(num_runs):
        t0 = time.perf_counter()

        # 1. Preprocessing
        img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
        )

        t1 = time.perf_counter()

        # 2. Inference
        outputs = session.run(None, {input_name: blob})

        t2 = time.perf_counter()

        # 3. Postprocessing
        boxes, confs, class_ids = postprocess(outputs[0], orig_shape, ratio, pad)

        t3 = time.perf_counter()

        preprocess_times.append((t1 - t0) * 1000.0)
        inference_times.append((t2 - t1) * 1000.0)
        postprocess_times.append((t3 - t2) * 1000.0)
        total_times.append((t3 - t0) * 1000.0)

        final_boxes, final_confs, final_class_ids = boxes, confs, class_ids

    # Output detection summary
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"Total Detections: {len(final_boxes)}")
    for i, (box, conf, cls) in enumerate(zip(final_boxes, final_confs, final_class_ids)):
        print(f"  Det #{i + 1}: Class={cls}, Conf={conf:.4f}, Box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    # Output timing summary
    def stats(data):
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "min": np.min(data),
            "max": np.max(data),
        }

    prep_s = stats(preprocess_times)
    inf_s = stats(inference_times)
    post_s = stats(postprocess_times)
    tot_s = stats(total_times)

    print("\n" + "=" * 60)
    print(f"TIMING BREAKDOWN ({num_runs} RUNS)")
    print("=" * 60)
    header = f"{'Phase':<15} | {'Avg (ms)':<10} | {'Median':<10} | {'Min':<8} | {'Max':<8} | {'Std':<8} | {'% Total':<8}"
    print(header)
    print("-" * len(header))

    tot_mean = tot_s["mean"]
    for name, s in [("Preprocess", prep_s), ("Inference", inf_s), ("Postprocess", post_s)]:
        pct = (s["mean"] / tot_mean) * 100.0 if tot_mean > 0 else 0
        print(
            f"{name:<15} | {s['mean']:<10.3f} | {s['median']:<10.3f} | {s['min']:<8.3f} | {s['max']:<8.3f} | {s['std']:<8.3f} | {pct:<7.1f}%"
        )

    print("-" * len(header))
    print(
        f"{'Total Frame':<15} | {tot_s['mean']:<10.3f} | {tot_s['median']:<10.3f} | {tot_s['min']:<8.3f} | {tot_s['max']:<8.3f} | {tot_s['std']:<8.3f} | {'100.0%':<8}"
    )
    print("=" * 60)
    print(f"Effective Throughput: {1000.0 / tot_s['mean']:.2f} FPS (Inference-only: {1000.0 / inf_s['mean']:.2f} FPS)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
