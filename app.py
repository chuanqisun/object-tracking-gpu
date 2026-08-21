import os
import tempfile
import time
from dataclasses import dataclass

import av
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.elements.image as st_image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from supervision import ByteTrack, Detections

# --- 1. AMD iGPU & ROCm / MIGraphX Configuration ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

# --- 2. Streamlit Compatibility Patch ---
@dataclass
class LayoutConfig:
    width: int = None

try:
    import streamlit.elements.lib.image_utils as iu

    def patched_image_to_url(
        image,
        width=None,
        clamp=False,
        channels="RGB",
        output_format="PNG",
        image_id="",
    ):
        config = LayoutConfig(width=width if isinstance(width, int) else None)
        return iu.image_to_url(
            image=image,
            layout_config=config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    st_image.image_to_url = patched_image_to_url
except Exception:
    pass

# --- 3. Page Configuration ---
st.set_page_config(
    page_title="YOLO26 iGPU Fast Instance Segmentation",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ YOLO26 Instance Segmentation & Tracking (AMD iGPU / ONNX Runtime)")
st.markdown(
    "High-performance real-time instance segmentation accelerated on **AMD Radeon iGPU (RDNA 3.5)** "
    "using **ONNX Runtime (MIGraphX FP16)** and **ByteTrack**."
)

INPUT_SIZE = 640

# --- Standard COCO Class Names for YOLO Segmentation ---
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# --- 4. Cached Session Loader ---
@st.cache_resource
def load_onnx_session(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"ONNX model not found at `{model_path}`. Please verify model export path.")
        st.stop()

    providers = [
        (
            "MIGraphXExecutionProvider",
            {
                "device_id": 0,
                "migraphx_fp16_enable": True,
                "migraphx_exhaustive_tune": False,
            },
        ),
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name

# Provide fallback paths for ONNX models
MODEL_PATHS = ["models/yolo26s-seg.onnx", "puck-eye-seg-s.onnx", "yolo26s-seg.onnx"]
selected_model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), MODEL_PATHS[0])

try:
    session, input_name = load_onnx_session(selected_model_path)
except Exception as e:
    st.error(f"Error loading ONNX Session: {e}")
    st.stop()


# --- 5. High-Performance Pre/Post-processing ---
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


def process_mask(protos, mask_coeffs, bboxes, orig_shape):
    """Vectorized sigmoid prototype matrix multiplication with bounding-box cropping."""
    c, mh, mw = protos.shape
    masks = 1 / (1 + np.exp(-np.matmul(mask_coeffs, protos.reshape(c, -1))))
    masks = masks.reshape(-1, mh, mw)

    scaled_masks = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.astype(int)
        mx1 = max(0, int(x1 * (mw / orig_shape[1])))
        my1 = max(0, int(y1 * (mh / orig_shape[0])))
        mx2 = min(mw, int(x2 * (mw / orig_shape[1])))
        my2 = min(mh, int(y2 * (mh / orig_shape[0])))

        mask = masks[i]
        cropped_mask = np.zeros_like(mask)
        cropped_mask[my1:my2, mx1:mx2] = mask[my1:my2, mx1:mx2]

        full_mask = cv2.resize(
            cropped_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        scaled_masks.append(full_mask > 0.5)

    return np.array(scaled_masks) if len(scaled_masks) > 0 else np.empty((0, *orig_shape), dtype=bool)


def nms_fast(boxes, scores, iou_threshold):
    """Greedy Non-Maximum Suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def run_igpu_inference(frame_bgr, conf_thresh=0.35, iou_thresh=0.45, class_indices=None):
    orig_shape = frame_bgr.shape[:2]

    # Preprocessing
    img, ratio, pad = letterbox(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
    blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    # ONNX Runtime Inference on iGPU
    outputs = session.run(None, {input_name: blob})

    p = np.squeeze(outputs[0])
    if p.ndim == 2 and p.shape[0] < p.shape[1]:
        p = p.T

    boxes = p[:, :4]
    num_classes = len(COCO_CLASSES)
    scores = p[:, 4 : 4 + num_classes]
    mask_coeffs = p[:, 4 + num_classes :]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    keep = confidences > conf_thresh
    if class_indices is not None and len(class_indices) > 0:
        cls_mask = np.isin(class_ids, class_indices)
        keep = keep & cls_mask

    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), np.empty((0, *orig_shape), dtype=bool)

    # Box conversion (cx, cy, w, h -> xyxy) & Letterbox inversion
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    xyxy = np.column_stack([x1, y1, x2, y2])

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_shape[1])
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_shape[0])

    # NMS
    nms_idx = nms_fast(xyxy, confidences, iou_thresh)
    xyxy = xyxy[nms_idx]
    confidences = confidences[nms_idx]
    class_ids = class_ids[nms_idx]
    mask_coeffs = mask_coeffs[nms_idx]

    protos = np.squeeze(outputs[1])
    masks = process_mask(protos, mask_coeffs, xyxy, orig_shape)

    return xyxy, confidences, class_ids, masks


def render_detections(
    frame,
    xyxy,
    confidences,
    class_ids,
    masks=None,
    track_ids=None,
    show_masks=True,
    show_boxes=True,
    show_labels=True,
    show_conf=True,
):
    out_frame = frame.copy()
    num_det = len(xyxy)

    # Color palette
    colors = [
        (0, 255, 120), (0, 165, 255), (255, 100, 0), (255, 0, 128),
        (0, 200, 255), (200, 255, 0), (128, 0, 255), (255, 255, 0),
    ]

    for i in range(num_det):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = map(int, xyxy[i])
        cls_id = int(class_ids[i])
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"Class {cls_id}"

        # 1. Overlay Mask
        if show_masks and masks is not None and i < len(masks):
            mask = masks[i]
            colored_mask = np.zeros_like(out_frame, dtype=np.uint8)
            colored_mask[mask] = color
            out_frame = cv2.addWeighted(out_frame, 1.0, colored_mask, 0.4, 0)

        # 2. Draw Box
        if show_boxes:
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)

        # 3. Draw Label
        if show_labels:
            label_parts = []
            if track_ids is not None and i < len(track_ids):
                label_parts.append(f"ID #{track_ids[i]}")
            label_parts.append(cls_name)
            if show_conf and i < len(confidences):
                label_parts.append(f"{confidences[i]:.2f}")

            label_text = " - ".join(label_parts)
            cv2.putText(
                out_frame,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    return out_frame


@st.cache_data
def get_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available if available else [0]


# --- 6. Sidebar Configuration ---
st.sidebar.header("⚙️ iGPU Inference Settings")
mode = st.sidebar.selectbox(
    "Select Operating Mode",
    ["Upload Image", "Upload Video", "Live Webcam (Direct)", "Live WebRTC Stream"],
)

st.sidebar.subheader("Confidence & IoU Thresholds")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

selected_classes = st.sidebar.multiselect(
    "Filter Classes (Leave empty to detect all)",
    options=COCO_CLASSES,
    default=[],
)
class_indices = [COCO_CLASSES.index(c) for c in selected_classes] if selected_classes else None

st.sidebar.subheader("Visualization Settings")
show_masks = st.sidebar.checkbox("Show Instance Masks", value=True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)


# --- 7. Application Modes ---

if mode == "Upload Image":
    st.header("🖼️ Image Instance Segmentation (iGPU Accelerated)")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("Executing ONNX Runtime inference on AMD iGPU..."):
            t0 = time.perf_counter()
            xyxy, confs, cls_ids, masks = run_igpu_inference(
                img_bgr, conf_thresh=conf_thresh, iou_thresh=iou_thresh, class_indices=class_indices
            )
            inf_time = (time.perf_counter() - t0) * 1000.0

            plotted_bgr = render_detections(
                img_bgr,
                xyxy,
                confs,
                cls_ids,
                masks=masks,
                show_masks=show_masks,
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_conf=show_conf,
            )
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.subheader(f"Segmented Output ({inf_time:.1f} ms)")
            st.image(plotted_rgb, use_container_width=True)

        num_instances = len(xyxy)
        st.subheader(f"📊 Detected Instances: {num_instances}")
        if num_instances > 0:
            data = []
            for idx in range(num_instances):
                cls_id = int(cls_ids[idx])
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"Class {cls_id}"
                data.append(
                    {
                        "Instance": idx + 1,
                        "Class": cls_name,
                        "Confidence": f"{confs[idx]:.2f}",
                        "Bounding Box [x1, y1, x2, y2]": str([round(x, 1) for x in xyxy[idx]]),
                        "Mask Area (Pixels)": int(masks[idx].sum()) if len(masks) > idx else 0,
                    }
                )
            st.dataframe(pd.DataFrame(data), use_container_width=True)


elif mode == "Upload Video":
    st.header("🎥 Video Instance Segmentation & ByteTrack (iGPU)")
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    col_v1, col_v2 = st.columns(2)
    frame_skip = col_v1.slider("Process every Nth frame", 1, 10, 1)
    use_tracking = col_v2.checkbox("Enable ByteTrack Tracking", value=True)

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file.")
        else:
            col_b1, col_b2 = st.columns([1, 4])
            start_btn = col_b1.button("🎬 Start Video Inference", type="primary")

            if start_btn:
                tracker = ByteTrack(track_activation_threshold=conf_thresh, lost_track_buffer=30, frame_rate=30)
                frame_placeholder = st.empty()
                m1, m2 = st.columns(2)
                fps_metric = m1.empty()
                count_metric = m2.empty()

                frame_count = 0
                while cap.isOpened():
                    t0 = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    xyxy, confs, cls_ids, masks = run_igpu_inference(
                        frame, conf_thresh=conf_thresh, iou_thresh=iou_thresh, class_indices=class_indices
                    )

                    track_ids = None
                    if use_tracking and len(xyxy) > 0:
                        detections = Detections(
                            xyxy=xyxy,
                            confidence=confs,
                            class_id=cls_ids,
                            mask=masks if len(masks) > 0 else None,
                        )
                        tracked = tracker.update_with_detections(detections)
                        xyxy = tracked.xyxy
                        cls_ids = tracked.class_id
                        track_ids = tracked.tracker_id
                        masks = tracked.mask

                    plotted_bgr = render_detections(
                        frame,
                        xyxy,
                        confs,
                        cls_ids,
                        masks=masks,
                        track_ids=track_ids,
                        show_masks=show_masks,
                        show_boxes=show_boxes,
                        show_labels=show_labels,
                        show_conf=show_conf,
                    )
                    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

                    fps = 1.0 / max(time.perf_counter() - t0, 1e-5)
                    frame_placeholder.image(plotted_rgb, channels="RGB", use_container_width=True)
                    fps_metric.metric("iGPU FPS", f"{fps:.1f}")
                    count_metric.metric("Active Instances", f"{len(xyxy)}")

                cap.release()
                st.success("Video processing completed.")


elif mode == "Live Webcam (Direct)":
    st.header("📹 Direct Webcam Tracking (AMD iGPU)")
    available_cams = get_available_cameras()
    selected_cam = st.selectbox("Select Webcam Device", options=available_cams, format_func=lambda x: f"Webcam Device {x}")
    use_tracking = st.checkbox("Enable ByteTrack Persistence", value=True)

    frame_placeholder = st.empty()
    col_f, col_c = st.columns(2)
    fps_metric = col_f.empty()
    count_metric = col_c.empty()

    col_s1, col_s2 = st.columns(2)
    start_tracking = col_s1.button("🚀 Start Live iGPU Inference", type="primary", use_container_width=True)
    stop_tracking = col_s2.button("⏹️ Stop Tracking", use_container_width=True)

    if start_tracking:
        cap = cv2.VideoCapture(selected_cam)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        tracker = ByteTrack(track_activation_threshold=conf_thresh, lost_track_buffer=30, frame_rate=30)

        if not cap.isOpened():
            st.error(f"Could not open camera device {selected_cam}.")
        else:
            while cap.isOpened():
                if stop_tracking:
                    break

                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab camera frame.")
                    break

                xyxy, confs, cls_ids, masks = run_igpu_inference(
                    frame, conf_thresh=conf_thresh, iou_thresh=iou_thresh, class_indices=class_indices
                )

                track_ids = None
                if use_tracking and len(xyxy) > 0:
                    detections = Detections(
                        xyxy=xyxy,
                        confidence=confs,
                        class_id=cls_ids,
                        mask=masks if len(masks) > 0 else None,
                    )
                    tracked = tracker.update_with_detections(detections)
                    xyxy = tracked.xyxy
                    cls_ids = tracked.class_id
                    track_ids = tracked.tracker_id
                    masks = tracked.mask

                plotted_bgr = render_detections(
                    frame,
                    xyxy,
                    confs,
                    cls_ids,
                    masks=masks,
                    track_ids=track_ids,
                    show_masks=show_masks,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_conf=show_conf,
                )
                plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

                fps = 1.0 / max(time.perf_counter() - t0, 1e-5)
                frame_placeholder.image(plotted_rgb, channels="RGB", use_container_width=True)
                fps_metric.metric("iGPU FPS", f"{fps:.1f}")
                count_metric.metric("Active Instances", f"{len(xyxy)}")

            cap.release()
            st.info("Direct tracking session ended.")


elif mode == "Live WebRTC Stream":
    st.header("🌐 WebRTC Browser Camera Streaming (iGPU Backend)")
    st.markdown("Direct in-browser camera streaming processed via ONNX Runtime GPU Acceleration.")

    class ONNXRTSegmentationVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.tracker = ByteTrack(track_activation_threshold=conf_thresh, lost_track_buffer=30, frame_rate=30)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            xyxy, confs, cls_ids, masks = run_igpu_inference(
                img,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                class_indices=class_indices,
            )

            track_ids = None
            if len(xyxy) > 0:
                detections = Detections(
                    xyxy=xyxy,
                    confidence=confs,
                    class_id=cls_ids,
                    mask=masks if len(masks) > 0 else None,
                )
                tracked = self.tracker.update_with_detections(detections)
                xyxy = tracked.xyxy
                cls_ids = tracked.class_id
                track_ids = tracked.tracker_id
                masks = tracked.mask

            plotted = render_detections(
                img,
                xyxy,
                confs,
                cls_ids,
                masks=masks,
                track_ids=track_ids,
                show_masks=show_masks,
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_conf=show_conf,
            )
            return av.VideoFrame.from_ndarray(plotted, format="bgr24")

    webrtc_streamer(
        key="yolo26-onnx-igpu-webrtc",
        video_processor_factory=ONNXRTSegmentationVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )