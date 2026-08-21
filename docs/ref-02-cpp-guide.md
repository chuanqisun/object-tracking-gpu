# Validated C++ Migration Guide

## 1. Install one consistent AMD software stack

Use the ROCm release whose official compatibility matrix supports Ubuntu 26.04, Radeon 890M, and `gfx1150`. Install ROCm development files and MIGraphX from the same release.

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake ninja-build git pkg-config \
    libopencv-dev libeigen3-dev nlohmann-json3-dev \
    libasio-dev libboost-system-dev
```

Install ROCm and MIGraphX using AMD’s package-manager instructions. Verify:

```bash
/opt/rocm/bin/rocminfo | grep -m1 gfx
```

Expected:

```text
gfx1150
```

Do **not** set:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.0
```

The Radeon 890M has native `gfx1150` support in current ROCm releases. Also do not mix ROCm versions such as ROCm 7.14 packages with `/opt/rocm-7.2.4`.

Use:

```bash
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
```

PyTorch is not required on the C++ runtime machine.

---

## 2. Build ONNX Runtime with MIGraphX

The AMD Python wheel is not a complete C++ development SDK. Build the ONNX Runtime release validated for your ROCm release.

Example for the ROCm 7.14 compatibility combination:

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.23.2
git submodule update --init --recursive

./build.sh \
    --config Release \
    --parallel \
    --build_shared_lib \
    --skip_tests \
    --use_migraphx \
    --migraphx_home /opt/rocm \
    --cmake_generator Ninja

sudo cmake --install build/Linux/Release \
    --prefix /opt/onnxruntime-migraphx
```

Confirm that the installation contains:

```text
/opt/onnxruntime-migraphx/include/onnxruntime_cxx_api.h
/opt/onnxruntime-migraphx/lib/libonnxruntime.so
```

If the upstream tag does not build against the selected ROCm release, use AMD MIGraphX’s official ONNX Runtime build helper rather than mixing unrelated releases.

---

## 3. Verify or re-export the ONNX model

Your application expects an end-to-end segmentation export containing:

```text
detections: [1, N, 38]
prototypes: [1, 32, 160, 160]
```

Each detection row must be:

```text
x1, y1, x2, y2, confidence, class_id, 32 mask coefficients
```

For two classes, a raw output such as:

```text
[1, 38, 8400]
```

is ambiguous and normally represents raw predictions:

```text
4 box values + 2 class scores + 32 mask coefficients
```

It must **not** be parsed as end-to-end `xyxy/confidence/class_id`.

If necessary, export with integrated NMS:

```bash
yolo export \
    model=puck-eye-seg-s.pt \
    format=onnx \
    imgsz=640 \
    half=True \
    simplify=True \
    nms=True
```

Always inspect the actual input/output types and shapes at application startup. Do not hard-code output order.

---

## 4. Add the official ByteTrack C++ core

Vendor the tracking core from the MIT-licensed official repository:

```bash
git submodule add https://github.com/ifzhang/ByteTrack.git \
    third_party/ByteTrack
git submodule update --init --recursive
```

Extract or compile the C++ tracking components from the official deployment implementation:

```text
BYTETracker.cpp
STrack.cpp
kalmanFilter.cpp
lapjv.cpp
```

The tracker should accept:

```cpp
struct TrackDetection {
    cv::Rect2f box;   // x, y, width, height
    float score;
    int class_id;
};
```

Expose ByteTrack’s thresholds rather than retaining hard-coded YOLOX defaults.

To reproduce the existing Supervision configuration, use:

```text
activation/high threshold = 0.35
new-track threshold       = 0.45
secondary/low threshold   = 0.10
matching threshold        = 0.80
frame rate                = 60
lost-track buffer         = 30
effective max lost frames = 30 × 60 / 30 = 60
```

Keep detections down to `0.05` before the tracker, matching `app.py`. ByteTrack will decide which low-confidence detections participate in secondary association.

For parity with Supervision, do not reject associations solely because class IDs differ. Preserve or update the class label from the matched detection.

---

## 5. Create the project

```text
cpp_yolo26_tracker/
├── CMakeLists.txt
├── models/
│   ├── puck-eye-seg-s.onnx
│   └── migraphx_cache/
├── static/
│   └── index.html
├── src/
│   ├── main.cpp
│   ├── inference_engine.cpp
│   ├── inference_engine.hpp
│   ├── postprocess.cpp
│   ├── postprocess.hpp
│   └── bytetrack/
└── third_party/
    └── ByteTrack/
```

---

## 6. Configure CMake

```cmake
cmake_minimum_required(VERSION 3.22)
project(puck_eye_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime-migraphx"
    CACHE PATH "MIGraphX-enabled ONNX Runtime")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED)

add_executable(puck-eye-server
    src/main.cpp
    src/inference_engine.cpp
    src/postprocess.cpp
    src/bytetrack/BYTETracker.cpp
    src/bytetrack/STrack.cpp
    src/bytetrack/kalmanFilter.cpp
    src/bytetrack/lapjv.cpp
)

target_include_directories(puck-eye-server PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
    src
    src/bytetrack
)

target_link_libraries(puck-eye-server PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
    ${OpenCV_LIBS}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    Threads::Threads
)

set_target_properties(puck-eye-server PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;/opt/rocm/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib;/opt/rocm/lib"
)
```

Do not link `migraphx_c` directly when inference is performed through ONNX Runtime. ONNX Runtime’s MIGraphX-enabled build owns the provider integration.

---

## 7. Initialize ONNX Runtime and MIGraphX

Set cache options before creating the session:

```cpp
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/migraphx/migraphx_provider_factory.h>

std::filesystem::create_directories("models/migraphx_cache");

setenv(
    "ORT_MIGRAPHX_CACHE_PATH",
    std::filesystem::absolute("models/migraphx_cache").c_str(),
    1);

setenv(
    "ORT_MIGRAPHX_MODEL_CACHE_PATH",
    std::filesystem::absolute("models/migraphx_cache").c_str(),
    1);

setenv("ORT_MIGRAPHX_FP16_ENABLE", "1", 1);

Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "puck-eye"};
Ort::SessionOptions options;

options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_ALL);

Ort::ThrowOnError(
    OrtSessionOptionsAppendExecutionProvider_MIGraphX(options, 0));

Ort::Session session{
    env,
    "models/puck-eye-seg-s.onnx",
    options
};
```

Do not use the removed variables:

```text
ORT_MIGRAPHX_LOAD_COMPILED_MODEL
ORT_MIGRAPHX_SAVE_COMPILED_MODEL
```

MIGraphX manages the model cache automatically.

Print available providers and fail if MIGraphX is missing:

```cpp
const auto providers = Ort::GetAvailableProviders();

if (std::find(
        providers.begin(),
        providers.end(),
        "MIGraphXExecutionProvider") == providers.end()) {
    throw std::runtime_error(
        "MIGraphXExecutionProvider is unavailable");
}
```

---

## 8. Implement preprocessing

Decode the WebSocket payload:

```cpp
std::vector<uchar> encoded(data.begin(), data.end());
cv::Mat frame = cv::imdecode(encoded, cv::IMREAD_COLOR);

if (frame.empty()) {
    return;
}
```

Implement letterboxing with the same rounding as Python:

```cpp
ratio = std::min(
    640.0F / static_cast<float>(frame.rows),
    640.0F / static_cast<float>(frame.cols));

int resized_w = std::lround(frame.cols * ratio);
int resized_h = std::lround(frame.rows * ratio);

float pad_x = (640 - resized_w) / 2.0F;
float pad_y = (640 - resized_h) / 2.0F;

int left   = std::lround(pad_x - 0.1F);
int right  = std::lround(pad_x + 0.1F);
int top    = std::lround(pad_y - 0.1F);
int bottom = std::lround(pad_y + 0.1F);
```

Create the NCHW tensor:

```cpp
cv::Mat blob = cv::dnn::blobFromImage(
    letterboxed,
    1.0 / 255.0,
    cv::Size(640, 640),
    cv::Scalar(),
    true,
    false,
    CV_32F);

std::array<int64_t, 4> shape{1, 3, 640, 640};

auto memory = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator,
    OrtMemTypeDefault);

Ort::Value input = Ort::Value::CreateTensor<float>(
    memory,
    blob.ptr<float>(),
    blob.total(),
    shape.data(),
    shape.size());
```

First verify whether the model input is FP32 or FP16. If it is FP16, explicitly convert and create an `Ort::Float16_t` tensor.

---

## 9. Parse end-to-end detections

Identify the detection output by shape rather than position. Require a rank-three tensor whose final dimension is 38.

For each row:

```cpp
float confidence = row[4];

if (confidence <= 0.05F) {
    continue;
}

int class_id = static_cast<int>(std::lround(row[5]));

float x1 = std::clamp(
    (row[0] - pad_x) / ratio, 0.0F,
    static_cast<float>(frame.cols));

float y1 = std::clamp(
    (row[1] - pad_y) / ratio, 0.0F,
    static_cast<float>(frame.rows));

float x2 = std::clamp(
    (row[2] - pad_x) / ratio, 0.0F,
    static_cast<float>(frame.cols));

float y2 = std::clamp(
    (row[3] - pad_y) / ratio, 0.0F,
    static_cast<float>(frame.rows));

TrackDetection detection{
    .box = cv::Rect2f{x1, y1, x2 - x1, y2 - y1},
    .score = confidence,
    .class_id = class_id
};
```

Reject invalid boxes:

```cpp
if (x2 <= x1 || y2 <= y1) {
    continue;
}
```

Do not run another NMS pass when the model contains integrated end-to-end NMS.

---

## 10. Skip unused segmentation masks

The current WebSocket response sends boxes, scores, classes and track IDs—but no masks. Therefore, do not calculate:

```text
mask coefficients × prototypes
```

This avoids a significant CPU cost.

Add mask processing later only if the frontend consumes masks. When implemented, use `cv::gemm` for all surviving detections in one operation and store box endpoints separately; never place `x2/y2` into `cv::Rect2f::width/height`.

---

## 11. Update one tracker per WebSocket connection

Each browser connection must own independent tracking state:

```cpp
struct SessionContext {
    ByteTracker tracker;
    std::mutex tracker_mutex;
    std::atomic_bool frame_in_flight{false};

    SessionContext()
        : tracker(
              60,     // frame rate
              30,     // base lost-track buffer
              0.35F,  // activation threshold
              0.45F,  // new-track threshold
              0.10F,  // low threshold
              0.80F)  // matching threshold
    {}
};
```

Processing sequence:

```cpp
std::vector<TrackDetection> detections =
    inference_engine.infer(frame);

std::vector<Track> tracks;
{
    std::lock_guard lock(context->tracker_mutex);
    tracks = context->tracker.update(detections);
}
```

Return active tracks directly:

```cpp
for (const auto& track : tracks) {
    const cv::Rect2f box = track.box();

    payload.push_back({
        {"box", {
            box.x,
            box.y,
            box.x + box.width,
            box.y + box.height
        }},
        {"score", track.score()},
        {"class_id", track.class_id()},
        {"track_id", track.id()}
    });
}
```

---

## 12. Use a bounded inference queue

Do not execute MIGraphX inference directly inside the WebSocket event-loop callback.

Use:

- one shared ONNX Runtime session;
- initially one inference worker;
- at most one active frame per connection;
- an optional single “latest pending frame” slot;
- stale-frame dropping.

This prevents browser clients from building an unbounded queue and preserves real-time tracking.

Start with one worker because unrestricted concurrent MIGraphX execution on an integrated GPU can increase latency. Add workers only after profiling.

---

## 13. Preserve the existing response format

```json
{
  "infer_ms": 12.4,
  "target_size": 640,
  "detections": [
    {
      "box": [10.0, 20.0, 60.0, 90.0],
      "score": 0.91,
      "class_id": 1,
      "track_id": 4
    }
  ]
}
```

Reset and destroy the tracker when the WebSocket closes.

---

## 14. Build and run

```bash
cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime-migraphx

cmake --build build
```

First compilation/tuning run:

```bash
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/onnxruntime-migraphx/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=1

./build/puck-eye-server
```

After the model cache is created:

```bash
export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0
./build/puck-eye-server
```

Do not inspect `.mxr` files or manually toggle deprecated load/save variables.

---

## 15. Validate parity

Test the same recorded sequence through Python and C++ and compare:

1. Detection output shapes and types.
2. Box coordinates within approximately one pixel.
3. Confidence and class values.
4. New-track activation near `0.45`.
5. Secondary association down to `0.10`.
6. Track loss after approximately 60 frames at 60 FPS.
7. Track IDs and ID switches.
8. Per-client tracker isolation.
9. Cache reuse after process restart.

Official references:

- [AMD ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [ONNX Runtime MIGraphX Execution Provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
- [ONNX Runtime EP build instructions](https://onnxruntime.ai/docs/build/eps.html)
- [Ultralytics ONNX export](https://docs.ultralytics.com/modes/export/)
- [Official ByteTrack repository](https://github.com/ifzhang/ByteTrack)
