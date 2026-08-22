import onnx
from collections import Counter
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

model = onnx.load("models/puck-eye-seg-s.onnx")
ops = Counter(n.op_type for n in model.graph.node)

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
print(session.get_inputs())
print(session.get_outputs())
print(ops)

for op in ("NonMaxSuppression", "NonZero", "TopK", "Where"):
    print(op, ops[op])