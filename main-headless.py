import asyncio
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ==============================================================================
# Model Configuration
# ==============================================================================
NUM_CLASSES = 2  # Set to 80 for standard COCO or your custom class count
INPUT_SIZE = 640
CONF_THRESH = 0.40
IOU_THRESH = 0.40  # NMS IoU Threshold
EXIT_ON_LOADED = False # Once the model is loaded, exit the script rather than serve it

# MODEL_PATH = "models/puck-eye-seg-s-nms.onnx" # nms=True, end2end=False
# MODEL_PATH = "models/puck-eye-seg-s.onnx" # nms=False, end2end=False
# MODEL_PATH = "models/puck-eye-seg-s-e2e-det10.onnx" # nms=False, end2end=True, det10=True
MODEL_PATH = "models/puck-eye-seg-s-e2e.onnx" # nms=False, end2end=True
# MODEL_PATH = "models/puck-eye-seg-s-e2e-tuned.onnx" # nms=False, end2end=True, exhaustive tuning
# ==============================================================================

# 1. Target AMD Radeon 890M (RDNA 3.5 / gfx1150)
os.environ["ROCM_PATH"] = "/opt/rocm"
# os.environ["MIGRAPHX_ENABLE_MLIR"] = "0"

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
    "migraphx_exhaustive_tune": False,
    # "migraphx_exhaustive_tune": not is_cached,
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
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

# Perform warmup inference pass to ensure the model is fully compiled, cached, and initialized
dummy_input = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
session.run(None, {input_name: dummy_input})
print(f"Model loaded and warmed up successfully: {MODEL_PATH}")

if EXIT_ON_LOADED:
    print("EXIT_ON_LOADED is True. Exiting script after model load.")
    sys.exit(0)


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


def postprocess(
    preds, orig_shape, ratio, pad, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH, num_classes=NUM_CLASSES
):
    p = np.squeeze(preds)
    if p.ndim == 1:
        p = np.expand_dims(p, axis=0)
    elif p.shape[0] < p.shape[1] and p.shape[1] > 100:
        p = p.T

    # YOLO26 exports end-to-end detections as xyxy, confidence, class_id, mask_coeffs.
    boxes = p[:, :4]
    confidences = p[:, 4]
    class_ids = p[:, 5].astype(np.int32)

    # 1. Confidence filtering
    keep = confidences > conf_thresh
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    # 2. Batched NMS filtering (using OpenCV's C++ cv2.dnn.NMSBoxesBatched)
    boxes_wh = boxes.copy()
    boxes_wh[:, 2] -= boxes_wh[:, 0]
    boxes_wh[:, 3] -= boxes_wh[:, 1]
    nms_indices = cv2.dnn.NMSBoxesBatched(boxes_wh, confidences, class_ids, 0.0, iou_thresh)

    if len(nms_indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    nms_indices = np.array(nms_indices).flatten()
    boxes = boxes[nms_indices]
    confidences = confidences[nms_indices]
    class_ids = class_ids[nms_indices]

    # Undo padding
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

    return xyxy, confidences, class_ids


def process_frame(frame_bytes: bytes) -> tuple[dict | None, dict]:
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
    boxes, confs, class_ids = postprocess(
        outputs[0], orig_shape, ratio, pad, conf_thresh=CONF_THRESH
    )
    t4 = time.perf_counter()

    detections_payload = []

    if len(boxes) > 0:
        for box, score, cls in zip(boxes, confs, class_ids):
            detections_payload.append({
                "box": [float(x) for x in box],
                "score": float(score),
                "class_id": int(cls),
            })
    t5 = time.perf_counter()

    infer_ms = (t5 - t0) * 1000.0

    metrics = {
        "t_decode": (t1 - t0) * 1000.0,
        "t_prep": (t2 - t1) * 1000.0,
        "t_inf": (t3 - t2) * 1000.0,
        "t_post": (t4 - t3) * 1000.0,
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

    frame_count = 0
    t_decode_list = []
    t_prep_list = []
    t_inf_list = []
    t_post_list = []
    t_total_list = []

    try:
        while True:
            # Receive binary frame payload (JPEG/PNG)
            frame_bytes = await websocket.receive_bytes()
            if not frame_bytes:
                continue

            response, metrics = await asyncio.to_thread(process_frame, frame_bytes)
            if response is None:
                continue

            if metrics:
                t_decode_list.append(metrics["t_decode"])
                t_prep_list.append(metrics["t_prep"])
                t_inf_list.append(metrics["t_inf"])
                t_post_list.append(metrics["t_post"])
                t_total_list.append(metrics["t_total"])
                frame_count += 1

                if frame_count % 30 == 0:
                    n = 30
                    print(
                        f"[Telemetry last 30 frames] Decode: {np.mean(t_decode_list[-n:]):.2f}ms | "
                        f"Prep: {np.mean(t_prep_list[-n:]):.2f}ms | "
                        f"Inf: {np.mean(t_inf_list[-n:]):.2f}ms | "
                        f"Post: {np.mean(t_post_list[-n:]):.2f}ms | "
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