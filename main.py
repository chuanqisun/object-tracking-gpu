import os
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from supervision import ByteTrack, Detections

# ==============================================================================
# Model Configuration
# ==============================================================================
NUM_CLASSES = 2  # Set to 80 for standard COCO or your custom class count
NUM_MASK_COEFFS = 32  # Standard YOLO prototype mask coefficients
INPUT_SIZE = 640
CONF_THRESH = 0.40
IOU_THRESH = 0.45  # NMS IoU Threshold
# ==============================================================================

# 1. Target AMD Radeon 890M (RDNA 3.5 / gfx1150)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.0"
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["MIGRAPHX_ENABLE_MLIR"] = "0"

# 2. MIGraphX Cache Configuration
cache_dir = "models/migraphx_cache"
os.makedirs(cache_dir, exist_ok=True)

is_cached = any(f.endswith(".mxr") for f in os.listdir(cache_dir))
cache_dir_abs = os.path.abspath(cache_dir)
os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = cache_dir_abs
os.environ["ORT_MIGRAPHX_CACHE_PATH"] = cache_dir_abs

migraphx_options = {
    "device_id": 0,
    "migraphx_fp16_enable": True,
    # Set to True on first compile to find fastest kernel variants on 890M
    "migraphx_exhaustive_tune": not is_cached,
}

if is_cached:
    print(f"Loading compiled MIGraphX cache from {cache_dir}...")
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
else:
    print(f"Compiling optimized MIGraphX kernels to {cache_dir}...")
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"

providers = [("MIGraphXExecutionProvider", migraphx_options)]
session = ort.InferenceSession("models/puck-eye-seg-s.onnx", providers=providers)
input_name = session.get_inputs()[0].name

# 3. Initialize ByteTrack
tracker = ByteTrack(
    track_activation_threshold=0.35, lost_track_buffer=30, frame_rate=60
)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Aspect-ratio preserving padding."""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


def process_mask_fast(protos, mask_coeffs, bboxes, orig_shape, pad, ratio):
    """
    Optimized mask processing:
    Computes masks directly on letterboxed prototype dimensions and maps to orig_shape.
    """
    c, mh, mw = protos.shape
    # Matrix multiply: (N, 32) @ (32, mh * mw) -> (N, mh, mw)
    masks = np.matmul(mask_coeffs, protos.reshape(c, -1))
    # In-place sigmoid
    masks = 1 / (1 + np.exp(-masks))
    masks = masks.reshape(-1, mh, mw)

    scaled_masks = []
    pad_w, pad_h = pad
    for i, box in enumerate(bboxes):
        mask = masks[i]

        # Calculate bounding box coordinates on prototype mask resolution (160x160)
        # 1. Map orig_shape box back to letterboxed 640x640 space
        x1_pad = box[0] * ratio + pad_w
        y1_pad = box[1] * ratio + pad_h
        x2_pad = box[2] * ratio + pad_w
        y2_pad = box[3] * ratio + pad_h

        # 2. Scale 640x640 space down to prototype space (mh, mw)
        scale_x = mw / INPUT_SIZE
        scale_y = mh / INPUT_SIZE
        mx1 = max(0, int(x1_pad * scale_x))
        my1 = max(0, int(y1_pad * scale_y))
        mx2 = min(mw, int(x2_pad * scale_x))
        my2 = min(mh, int(y2_pad * scale_y))

        # Crop outside of box
        cropped_mask = np.zeros_like(mask)
        cropped_mask[my1:my2, mx1:mx2] = mask[my1:my2, mx1:mx2]

        # Un-pad and resize only to original shape
        unpad_h = int(round(orig_shape[0] * ratio * scale_y))
        unpad_w = int(round(orig_shape[1] * ratio * scale_x))
        pad_top = int(round(pad_h * scale_y))
        pad_left = int(round(pad_w * scale_x))

        mask_unpad = cropped_mask[pad_top : pad_top + unpad_h, pad_left : pad_left + unpad_w]
        if mask_unpad.size == 0:
            full_mask = np.zeros(orig_shape, dtype=bool)
        else:
            full_mask = cv2.resize(
                mask_unpad, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR
            ) > 0.5
        scaled_masks.append(full_mask)

    return np.array(scaled_masks) if len(scaled_masks) > 0 else np.empty((0, *orig_shape), dtype=bool)


def postprocess(
    preds, protos, orig_shape, ratio, pad, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH, num_classes=NUM_CLASSES
):
    p = np.squeeze(preds)
    if p.shape[0] < p.shape[1]:
        p = p.T

    protos = np.squeeze(protos)
    num_protos = protos.shape[0]

    # YOLO26 exports end-to-end detections as xyxy, confidence, class_id, masks.
    boxes = p[:, :4]
    confidences = p[:, 4]
    class_ids = p[:, 5].astype(np.int32)
    mask_coeffs = p[:, 6 : 6 + num_protos]

    # 1. Confidence filtering
    keep = confidences > conf_thresh
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty((0, *orig_shape))

    # The export already applied confidence filtering and NMS; only undo padding.
    x1 = (boxes[:, 0] - pad[0]) / ratio
    y1 = (boxes[:, 1] - pad[1]) / ratio
    x2 = (boxes[:, 2] - pad[0]) / ratio
    y2 = (boxes[:, 3] - pad[1]) / ratio

    xyxy = np.column_stack([
        np.clip(x1, 0, orig_shape[1]),
        np.clip(y1, 0, orig_shape[0]),
        np.clip(x2, 0, orig_shape[1]),
        np.clip(y2, 0, orig_shape[0]),
    ])

    # Compute segmentation masks only on confidence-surviving boxes.
    masks = process_mask_fast(protos, mask_coeffs, xyxy, orig_shape, pad, ratio)

    return xyxy, confidences, class_ids, masks


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps_window = deque(maxlen=120)

    print("Running optimized YOLO-seg + ByteTrack on Radeon 890M iGPU...")

    while cap.isOpened():
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        orig_shape = frame.shape[:2]

        # Fast Preprocessing using cv2.dnn.blobFromImage
        img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
        )

        # iGPU Inference
        outputs = session.run(None, {input_name: blob})

        # Postprocessing + NMS
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

            # Combined Mask Overlay (Fast Vectorized Blend)
            if tracked.mask is not None and len(tracked.mask) > 0:
                combined_mask = np.any(tracked.mask, axis=0)
                mask_overlay = np.zeros_like(frame)
                mask_overlay[combined_mask] = (0, 165, 255)
                cv2.addWeighted(mask_overlay, 0.4, frame, 1.0, 0, dst=frame)

            for xyxy, track_id, cls in zip(
                tracked.xyxy, tracked.tracker_id, tracked.class_id
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

        fps_window.append(time.perf_counter())
        while fps_window and fps_window[-1] - fps_window[0] > 1.0:
            fps_window.popleft()

        if len(fps_window) > 1:
            avg_fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
        else:
            avg_fps = 0.0

        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f} (Radeon 890M)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLO-seg + ByteTrack", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()