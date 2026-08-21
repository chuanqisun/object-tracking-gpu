### Step 1: Project Setup with `uv`

Initialize a new `uv` project directory and target Python 3.12 (standard for modern ROCm/MIGraphX wheels):

```bash
# 1. Initialize project
mkdir yolo26_tracker && cd yolo26_tracker
uv init --app --python 3.12
```

Install vendor packages

- [ROCm](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pip)

```sh
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,device-gfx1151]==7.14.0"

sudo ln -s /opt/rocm-7.2.4 /opt/rocm
```

(to uninstall, you need to manually remove the symbolic link and uninstall the package)

- [MIGraphX](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/migraphx.html?i=pkgman)

```sh
wget https://rocm.frameworks.amd.com/whl-multi-arch/migraphx/migraphx-2.16.0%2Brocm7.14.0-cp312-none-manylinux_2_28_x86_64.whl
uv pip install migraphx-2.16.0+rocm7.14.0-cp312-none-manylinux_2_28_x86_64.whl
```

- [PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=linux&i=pip&w=compute&gpu=9-hx-370&gfx=gfx1150&pytorch-ver=2.12.0)

```sh
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1150]==2.12.0+rocm7.14.0" \
    "torchvision[device-gfx1150]==0.27.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0"
```

- [onnx for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html)

```sh
uv pip install https://rocm.frameworks.amd.com/whl-multi-arch/onnxruntime-migraphx/onnxruntime_migraphx-1.23.2%2Brocm7.14.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

```bash
# 2. Add standard runtime & tracking dependencies
uv pip install opencv-python supervision

# 3. Add rocm specific pytorch wheel (if needed)
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1150]==2.12.0+rocm7.14.0" \
    "torchvision[device-gfx1150]==0.27.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0"
```

---

### Step 2: Export `yolo26s-seg.pt` to FP16 ONNX

Run the export tool via `uv run` inside your project environment:

```bash
uv run yolo export model=yolo26s-seg.pt format=onnx imgsz=640 half=True simplify=True
```

_This produces `yolo26s-seg.onnx` in your current directory._

---

### Step 3: Complete Python Implementation (`main.py`)

Replace `main.py` with the following optimized implementation:

```python
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from supervision import ByteTrack, Detections

# 1. Target AMD Radeon 890M (RDNA 3.5 / gfx1150)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

# 2. Configure Execution Providers (Strictly iGPU FP16)
providers = [
    (
        "MIGraphXExecutionProvider",
        {
            "device_id": 0,
            "migraphx_fp16_enable": True,
            "migraphx_exhaustive_tune": False,
        },
    ),
    (
        "ROCMExecutionProvider",
        {
            "device_id": 0,
            "tunable_op_enable": 1,
            "tunable_op_tuning_enable": 0,
        },
    ),
]

# Initialize Session
session = ort.InferenceSession("yolo26s-seg.onnx", providers=providers)
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
```

---

### Step 4: Run the Application

Launch the script directly with `uv`:

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
uv run python main.py
```
