import asyncio
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
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


def process_frame(frame_bytes: bytes, session_tracker: ByteTrack) -> tuple[dict | None, dict]:
    t0 = time.perf_counter()

    # Decode image from binary buffer
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    if frame is None:
        return None, {}

    orig_shape = frame.shape[:2]

    # Fast Preprocessing using cv2.dnn.blobFromImage
    img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    t2 = time.perf_counter()

    # iGPU Inference
    outputs = session.run(None, {input_name: blob})
    t3 = time.perf_counter()

    # Postprocessing
    boxes, confs, class_ids, masks = postprocess(
        outputs[0], outputs[1], orig_shape, ratio, pad, conf_thresh=0.05
    )
    t4 = time.perf_counter()

    detections_payload = []

    if len(boxes) > 0:
        detections = Detections(
            xyxy=boxes,
            confidence=confs,
            class_id=class_ids,
            mask=masks if len(masks) > 0 else None,
        )

        tracked = session_tracker.update_with_detections(detections)

        for xyxy, score, cls, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            detections_payload.append({
                "box": [float(x) for x in xyxy],
                "score": float(score),
                "class_id": int(cls),
                "track_id": int(track_id),
            })
    t5 = time.perf_counter()

    infer_ms = (t5 - t0) * 1000.0

    metrics = {
        "t_decode": (t1 - t0) * 1000.0,
        "t_prep": (t2 - t1) * 1000.0,
        "t_inf": (t3 - t2) * 1000.0,
        "t_post": (t4 - t3) * 1000.0,
        "t_track": (t5 - t4) * 1000.0,
        "t_total": (t5 - t0) * 1000.0,
    }

    payload = {
        "infer_ms": infer_ms,
        "target_size": INPUT_SIZE,
        "detections": detections_payload,
    }

    return payload, metrics


app = FastAPI()


@app.get("/")
async def get_index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Reset tracking state for new client connection
    session_tracker = ByteTrack(
        track_activation_threshold=0.35, lost_track_buffer=30, frame_rate=60
    )

    frame_count = 0
    t_decode_list = []
    t_prep_list = []
    t_inf_list = []
    t_post_list = []
    t_track_list = []
    t_total_list = []

    try:
        while True:
            # Receive binary frame payload (JPEG/PNG)
            frame_bytes = await websocket.receive_bytes()
            if not frame_bytes:
                continue

            response, metrics = await asyncio.to_thread(process_frame, frame_bytes, session_tracker)
            if response is None:
                continue

            if metrics:
                t_decode_list.append(metrics["t_decode"])
                t_prep_list.append(metrics["t_prep"])
                t_inf_list.append(metrics["t_inf"])
                t_post_list.append(metrics["t_post"])
                t_track_list.append(metrics["t_track"])
                t_total_list.append(metrics["t_total"])
                frame_count += 1

                if frame_count % 30 == 0:
                    n = 30
                    print(
                        f"[Telemetry last 30 frames] Decode: {np.mean(t_decode_list[-n:]):.2f}ms | "
                        f"Prep: {np.mean(t_prep_list[-n:]):.2f}ms | "
                        f"Inf: {np.mean(t_inf_list[-n:]):.2f}ms | "
                        f"Post: {np.mean(t_post_list[-n:]):.2f}ms | "
                        f"Track: {np.mean(t_track_list[-n:]):.2f}ms | "
                        f"Total: {np.mean(t_total_list[-n:]):.2f}ms ({1000.0 / np.mean(t_total_list[-n:]):.1f} FPS)"
                    )

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


def main():
    port = int(os.environ.get("PORT", 18888))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting WebSocket server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()