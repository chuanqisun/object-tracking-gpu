**Yes, there is official C++ support for the entire stack.**

AMD and ONNX Runtime provide complete C++ APIs for:

1. **ONNX Runtime with MIGraphX Execution Provider** (`onnxruntime_cxx_api.h`).
2. **Native AMD MIGraphX C++ API** (alternative to ORT via `#include <migraphx/migraphx.hpp>`).
3. **OpenCV C++** (`#include <opencv2/opencv.hpp>`) for camera capture, preprocessing (letterbox/blob), matrix multiplications, mask rendering, and display.

---

### 1. Official Documentation & References

- **ONNX Runtime MIGraphX EP (C/C++ Documentation):**
  [ONNX Runtime AMD MIGraphX Execution Provider Guide](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
- **MIGraphX C++ API Reference:**
  [AMD ROCm MIGraphX Documentation](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)
- **ROCm Linux Installation & Header Setup:**
  [AMD ROCm Documentation](https://rocm.docs.amd.com/)

---

### 2. Setting Provider Options & Environment in C++

All the options you used in Python (`HSA_OVERRIDE_GFX_VERSION`, `MIGRAPHX_ENABLE_MLIR`, `ORT_MIGRAPHX_MODEL_CACHE_PATH`, `migraphx_fp16_enable`, `migraphx_exhaustive_tune`) translate directly to C++:

#### C++ Session Configuration

```cpp
#include <onnxruntime_cxx_api.h>
#include <cstdlib>

// Set GPU Target & Optimization flags in the process environment
setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.0", 1);
setenv("ROCM_PATH", "/opt/rocm", 1);
setenv("MIGRAPHX_ENABLE_MLIR", "0", 1);

// Caching and tuning options (can be passed via OrtSessionOptionsAppendExecutionProvider_MIGraphX
// or via the generic SessionOptions provider options map)
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Tracker");
Ort::SessionOptions session_options;

// Configure MIGraphX Execution Provider
OrtMIGraphXProviderOptions migraphx_options{};
migraphx_options.device_id = 0;
migraphx_options.migraphx_fp16_enable = 1;
migraphx_options.migraphx_exhaustive_tune = is_cached ? 0 : 1;
migraphx_options.migraphx_save_model_path = "models/migraphx_cache/compiled.mxr";
migraphx_options.migraphx_load_model_path = "models/migraphx_cache/compiled.mxr";

// Append the EP
session_options.AppendExecutionProvider_MIGraphX(migraphx_options);

// Load Model
Ort::Session session(env, "models/puck-eye-seg-s.onnx", session_options);
```

---

### 3. Complete C++ Implementation (`main.cpp`)

This implementation replaces `app.py` by performing preprocessing via OpenCV, running ONNX Runtime on the MIGraphX EP, computing the prototype segmentation masks, and tracking detections:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include <numeric>
#include <filesystem>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

constexpr int INPUT_SIZE = 640;
constexpr int NUM_CLASSES = 2;
constexpr int NUM_MASK_COEFFS = 32;
constexpr float CONF_THRESH = 0.40f;

struct LetterboxResult {
    cv::Mat image;
    float ratio;
    cv::Point2f pad;
};

LetterboxResult letterbox(const cv::Mat& src, cv::Size new_shape = {INPUT_SIZE, INPUT_SIZE}, cv::Scalar color = cv::Scalar(114, 114, 114)) {
    int shape_w = src.cols;
    int shape_h = src.rows;
    float r = std::min((float)new_shape.width / shape_w, (float)new_shape.height / shape_h);
    int new_unpad_w = std::round(shape_w * r);
    int new_unpad_h = std::round(shape_h * r);
    float dw = (new_shape.width - new_unpad_w) / 2.0f;
    float dh = (new_shape.height - new_unpad_h) / 2.0f;

    cv::Mat resized;
    if (src.cols != new_unpad_w || src.rows != new_unpad_h) {
        cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = src;
    }

    int top = std::round(dh - 0.1f);
    int bottom = std::round(dh + 0.1f);
    int left = std::round(dw - 0.1f);
    int right = std::round(dw + 0.1f);

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return {out, r, cv::Point2f(dw, dh)};
}

struct Detection {
    cv::Rect2f box;
    float score;
    int class_id;
    cv::Mat mask;
};

void process_mask_fast(const float* proto_data, int proto_c, int proto_h, int proto_w,
                       const float* mask_coeff, const cv::Rect2f& box,
                       const cv::Size& orig_shape, const cv::Point2f& pad, float ratio,
                       cv::Mat& out_mask) {
    // 1. Matrix multiplication: (1, 32) @ (32, proto_h * proto_w)
    cv::Mat coeffs(1, proto_c, CV_32F, const_cast<float*>(mask_coeff));
    cv::Mat protos(proto_c, proto_h * proto_w, CV_32F, const_cast<float*>(proto_data));
    cv::Mat mask_flat = coeffs * protos;

    // 2. Sigmoid activation
    cv::Mat mask_proto = mask_flat.reshape(1, proto_h);
    cv::exp(-mask_proto, mask_proto);
    mask_proto = 1.0f / (1.0f + mask_proto);

    // 3. Map orig_shape box back to letterboxed space and scale down to prototype space
    float scale_x = (float)proto_w / INPUT_SIZE;
    float scale_y = (float)proto_h / INPUT_SIZE;

    float x1_pad = box.x * ratio + pad.x;
    float y1_pad = box.y * ratio + pad.y;
    float x2_pad = (box.x + box.width) * ratio + pad.x;
    float y2_pad = (box.y + box.height) * ratio + pad.y;

    int mx1 = std::max(0, (int)(x1_pad * scale_x));
    int my1 = std::max(0, (int)(y1_pad * scale_y));
    int mx2 = std::min(proto_w, (int)(x2_pad * scale_x));
    int my2 = std::min(proto_h, (int)(y2_pad * scale_y));

    cv::Mat cropped_mask = cv::Mat::zeros(proto_h, proto_w, CV_32F);
    if (mx2 > mx1 && my2 > my1) {
        mask_proto(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1))
            .copyTo(cropped_mask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)));
    }

    // 4. Crop pad and resize back to original image shape
    int unpad_w = std::round(orig_shape.width * ratio * scale_x);
    int unpad_h = std::round(orig_shape.height * ratio * scale_y);
    int pad_left = std::round(pad.x * scale_x);
    int pad_top = std::round(pad.y * scale_y);

    pad_left = std::clamp(pad_left, 0, proto_w - 1);
    pad_top = std::clamp(pad_top, 0, proto_h - 1);
    unpad_w = std::clamp(unpad_w, 1, proto_w - pad_left);
    unpad_h = std::clamp(unpad_h, 1, proto_h - pad_top);

    cv::Mat mask_unpad = cropped_mask(cv::Rect(pad_left, pad_top, unpad_w, unpad_h));
    cv::Mat resized_mask;
    cv::resize(mask_unpad, resized_mask, orig_shape, 0, 0, cv::INTER_LINEAR);

    out_mask = (resized_mask > 0.5f);
}

int main() {
    // 1. Environmental Flags
    setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.0", 1);
    setenv("ROCM_PATH", "/opt/rocm", 1);
    setenv("MIGRAPHX_ENABLE_MLIR", "0", 1);

    std::string cache_dir = "models/migraphx_cache";
    fs::create_directories(cache_dir);
    bool is_cached = false;
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        if (entry.path().extension() == ".mxr") { is_cached = true; break; }
    }

    // Set cache environment variables
    std::string abs_cache = fs::canonical(cache_dir).string();
    setenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", abs_cache.c_str(), 1);
    setenv("ORT_MIGRAPHX_CACHE_PATH", abs_cache.c_str(), 1);
    setenv(is_cached ? "ORT_MIGRAPHX_LOAD_COMPILED_MODEL" : "ORT_MIGRAPHX_SAVE_COMPILED_MODEL", "1", 1);

    // 2. Initialize ONNX Runtime Session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Radeon890M_YOLO");
    Ort::SessionOptions session_options;

    // Use MIGraphX Provider
    OrtMIGraphXProviderOptions migraphx_options{};
    migraphx_options.device_id = 0;
    migraphx_options.migraphx_fp16_enable = 1;
    migraphx_options.migraphx_exhaustive_tune = is_cached ? 0 : 1;
    session_options.AppendExecutionProvider_MIGraphX(migraphx_options);

    const char* model_path = "models/puck-eye-seg-s.onnx";
    Ort::Session session(env, model_path, session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video device." << std::endl;
        return -1;
    }

    std::deque<std::chrono::high_resolution_clock::time_point> fps_window;
    cv::Mat frame;

    std::vector<int64_t> input_shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {"output0", "output1"};

    while (cap.read(frame)) {
        cv::Size orig_shape = frame.size();

        // Preprocessing
        LetterboxResult lb = letterbox(frame, {INPUT_SIZE, INPUT_SIZE});
        cv::Mat blob = cv::dnn::blobFromImage(lb.image, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), true, false);

        // Memory info for input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

        // Inference
        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

        float* pred_data = outputs[0].GetTensorMutableData<float>();
        float* proto_data = outputs[1].GetTensorMutableData<float>();

        auto pred_info = outputs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> pred_shape = pred_info.GetShape(); // [1, 38, 8400] or [1, 8400, 38]

        int num_boxes = (pred_shape[1] > pred_shape[2]) ? pred_shape[1] : pred_shape[2];
        int num_feats = (pred_shape[1] > pred_shape[2]) ? pred_shape[2] : pred_shape[1];

        std::vector<Detection> detections;
        cv::Mat combined_mask = cv::Mat::zeros(orig_shape, CV_8UC1);

        for (int i = 0; i < num_boxes; ++i) {
            float* row = pred_data + (i * num_feats);
            float score = row[4];
            if (score > CONF_THRESH) {
                int cls_id = static_cast<int>(row[5]);
                float x1 = (row[0] - lb.pad.x) / lb.ratio;
                float y1 = (row[1] - lb.pad.y) / lb.ratio;
                float x2 = (row[2] - lb.pad.x) / lb.ratio;
                float y2 = (row[3] - lb.pad.y) / lb.ratio;

                cv::Rect2f box(x1, y1, x2 - x1, y2 - y1);
                cv::Mat mask;
                process_mask_fast(proto_data, 32, 160, 160, row + 6, box, orig_shape, lb.pad, lb.ratio, mask);

                detections.push_back({box, score, cls_id, mask});
                cv::bitwise_or(combined_mask, mask, combined_mask);
            }
        }

        // Apply blended mask
        if (!detections.empty()) {
            cv::Mat mask_overlay = cv::Mat::zeros(frame.size(), frame.type());
            mask_overlay.setTo(cv::Scalar(0, 165, 255), combined_mask);
            cv::addWeighted(mask_overlay, 0.4, frame, 1.0, 0, frame);
        }

        // Draw Bounding Boxes
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 120), 2);
            cv::putText(frame, "Class " + std::to_string(det.class_id),
                        cv::Point(det.box.x, std::max(20.0f, det.box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 120), 2);
        }

        // FPS Calculation
        auto now = std::chrono::high_resolution_clock::now();
        fps_window.push_back(now);
        while (std::chrono::duration<float>(now - fps_window.front()).count() > 1.0f) {
            fps_window.pop_front();
        }
        float fps = (fps_window.size() > 1) ? (fps_window.size() - 1) / std::chrono::duration<float>(now - fps_window.front()).count() : 0.0f;

        cv::putText(frame, "FPS: " + std::to_string((int)fps) + " (Radeon 890M - C++)",
                    cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        cv::imshow("YOLO-seg C++ (MIGraphX)", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
```

---

### 4. Tracking with ByteTrack in C++

In Python, you used `supervision.ByteTrack`. Supervision does not maintain an official C++ binary, but ByteTrack is open-source C++ with official implementations available as drop-in single header/source files:

- Official ByteTrack C++ Implementation: [ByteTrack C++ (Github)](https://github.com/ifzhang/ByteTrack/tree/main/deploy/ByteTrack/cpp)

You can drop `BYTETracker.h` and `BYTETracker.cpp` directly into your CMake project without needing third-party wrappers.

---

### 5. `CMakeLists.txt` Build File

```cmake
cmake_minimum_required(VERSION 3.20)
project(yolo_migraphx_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# ROCm & ONNX Runtime paths
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
set(ORT_PATH "/usr/local" CACHE PATH "Path to ONNX Runtime MIGraphX installation")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_PATH}/include
    ${ROCM_PATH}/include
)

link_directories(
    ${ORT_PATH}/lib
    ${ROCM_PATH}/lib
)

add_executable(yolo_tracker main.cpp)

target_link_libraries(yolo_tracker
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    migraphx_c
)
```

## References

1. [onnxruntime.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG4oLZxuTHUVHjibiq2xeWaygffpL6i0lfVSGdESrMDDAb-vUoGa-HzgpvNDey4Zx61wqwuj_r8QlGFGcViImfqJzz7xBVcdO3qOkVNJ4V4HZ5UWdsHROJJLrdHNRjumm2IoUusaOmioGHEK3CvA9d_TkjcIZ9Bwsga8TrGY3X096-yH_m8)
2. [onnxruntime.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0uJ-pKOEGmg9x5G9HXA-Iebos8YCGUhbMnY-NCIBI30889Ws5rXwMSBJahRBWuZ3NXwjtKVOZNtYlfbKBtYTUnBEaPtVpgFdrwAUHD9lWkaxBsMU_xaocYxk93t6W40EbvYdhbsE=)
3. [amd.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG2Aa-iqtDzTxEXkCLf7Y2DfCOJqxECvPMfMuqASkkwnWdJTnHI1h4dfslNJKgbXijqdL9tu7r-2TkRByOJ9B-ROQXejIsHSC5zQzxDz6-4OUYYzVLhZ_XMC-TwZXF4QeJCqMhlszaEwOk8hn8g53gyjadepi-hlGbsB8SJxSMsnVEAReGcd5QbkxGWH2USeQcV9_SM_ibLwbMdTiqUZtE12qOO0-Q4uejB6Q==)
