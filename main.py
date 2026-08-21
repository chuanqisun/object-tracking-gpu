import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from supervision import ByteTrack, Detections


# 1. Target AMD Radeon 890M (RDNA 3.5 / gfx1150)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.0"
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["MIGRAPHX_ENABLE_MLIR"] = "0"

# 2. MIGraphX Cache Configuration (Requires a directory, not a file)
cache_dir = "models/migraphx_cache"
os.makedirs(cache_dir, exist_ok=True)

# Check if any precompiled .mxr files already exist in the cache directory
is_cached = any(f.endswith(".mxr") for f in os.listdir(cache_dir))

cache_dir_abs = os.path.abspath(cache_dir)
os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = cache_dir_abs
os.environ["ORT_MIGRAPHX_CACHE_PATH"] = cache_dir_abs

migraphx_options = {
    "device_id": 0,
    "migraphx_fp16_enable": True,
    "migraphx_exhaustive_tune": False,
}

if is_cached:
    print(f"Loading compiled MIGraphX cache from {cache_dir}...")
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
else:
    print(f"No compiled cache found. Compiling and saving to {cache_dir}...")
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"

providers = [("MIGraphXExecutionProvider", migraphx_options)]

session = ort.InferenceSession("models/puck-eye-seg-s.onnx", providers=providers)
input_name = session.get_inputs()[0].name
INPUT_SIZE = 640

# 3. Initialize ByteTrack
tracker = ByteTrack(
    track_activation_threshold=0.35, lost_track_buffer=30, frame_rate=30
)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Maintain aspect ratio with uniform padding."""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


def process_mask(protos, mask_coeffs, bboxes, shape):
    """Vectorized sigmoid and prototype matrix multiplication."""
    c, mh, mw = protos.shape
    masks = 1 / (1 + np.exp(-np.matmul(mask_coeffs, protos.reshape(c, -1))))
    masks = masks.reshape(-1, mh, mw)

    scaled_masks = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.astype(int)
        mx1 = max(0, int(x1 * (mw / shape[1])))
        my1 = max(0, int(y1 * (mh / shape[0])))
        mx2 = min(mw, int(x2 * (mw / shape[1])))
        my2 = min(mh, int(y2 * (mh / shape[0])))

        mask = masks[i]
        cropped_mask = np.zeros_like(mask)
        cropped_mask[my1:my2, mx1:mx2] = mask[my1:my2, mx1:mx2]

        full_mask = cv2.resize(
            cropped_mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
        )
        scaled_masks.append(full_mask > 0.5)

    return (
        np.array(scaled_masks) if len(scaled_masks) > 0 else np.empty((0, *shape))
    )


def postprocess(preds, protos, orig_shape, ratio, pad, conf_thresh=0.4):
    p = np.squeeze(preds)
    if p.shape[0] < p.shape[1]:
        p = p.T

    boxes = p[:, :4]  # cx, cy, w, h
    scores = p[:, 4:84]  # 80 classes
    mask_coeffs = p[:, 84:]  # 32 protos

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    keep = confidences > conf_thresh
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    if len(boxes) == 0:
        return (
            np.empty((0, 4)),
            np.empty(0),
            np.empty(0),
            np.empty((0, *orig_shape)),
        )

    # Box conversion & Letterbox inversion
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    xyxy = np.column_stack([x1, y1, x2, y2])

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_shape[1])
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_shape[0])

    protos = np.squeeze(protos)
    masks = process_mask(protos, mask_coeffs, xyxy, orig_shape)

    return xyxy, confidences, class_ids, masks


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Running YOLO26s-seg + ByteTrack on Radeon 890M iGPU...")

    while cap.isOpened():
        start_t = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        orig_shape = frame.shape[:2]

        # Preprocessing
        img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        # iGPU Inference
        outputs = session.run(None, {input_name: blob})

        # Postprocessing
        boxes, confs, class_ids, masks = postprocess(
            outputs[0], outputs[1], orig_shape, ratio, pad
        )

        if len(boxes) > 0:
            detections = Detections(
                xyxy=boxes,
                confidence=confs,
                class_id=class_ids,
                mask=masks if len(masks) > 0 else None,
            )

            tracked = tracker.update_with_detections(detections)

            for i, (xyxy, track_id, cls) in enumerate(
                zip(tracked.xyxy, tracked.tracker_id, tracked.class_id)
            ):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                cv2.putText(
                    frame,
                    f"ID #{track_id} - Class {cls}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 120),
                    2,
                )

                if tracked.mask is not None and i < len(tracked.mask):
                    mask = tracked.mask[i]
                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    colored_mask[mask] = (0, 165, 255)
                    frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

        fps = 1.0 / (time.perf_counter() - start_t)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} (Radeon 890M)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLO26s-seg + ByteTrack", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()