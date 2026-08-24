# Overview

![A hand holding an egg. The shape of the hand is marked with a wireframe that
indicates the identified
structure](https://developers.google.com/static/mediapipe/images/solutions/examples/hand_landmark.png)

The MediaPipe Hand Landmarker task lets you detect the landmarks of the hands in an image.
You can use this task to locate key points of hands and render visual effects on
them. This task operates on image data with a machine learning (ML) model as
static data or a continuous stream and outputs hand landmarks in image
coordinates, hand landmarks in world coordinates and handedness(left/right hand)
of multiple detected hands.

[Try it!](https://google-ai-edge.github.io/mediapipe-samples-web/#/vision/hand_landmarker)

## Get Started

Start using this task by following one of these implementation guides for your
target platform. These platform-specific guides walk you through a basic
implementation of this task, including a recommended model, and code example
with recommended configuration options:

- **Android** - [Code
  example](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/hand_landmarker/android)
  - [Guide](https://developers.google.com/edge/mediapipe/solutions/vision/android)
- **Python** - [Code
  example](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
  - [Guide](https://developers.google.com/edge/mediapipe/solutions/vision/python)
- **Web** - [Code example](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/hand-landmarker.ts) - [Guide](https://developers.google.com/edge/mediapipe/solutions/vision/web_js)

## Task details

This section describes the capabilities, inputs, outputs, and configuration
options of this task.

### Features

- **Input image processing** - Processing includes image rotation, resizing, normalization, and color space conversion.
- **Score threshold** - Filter results based on prediction scores.

| Task inputs                                                                                                                                        | Task outputs                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The Hand Landmarker accepts an input of one of the following data types: - Still images <!-- --> - Decoded video frames <!-- --> - Live video feed | The Hand Landmarker outputs the following results: - Handedness of detected hands <!-- --> - Landmarks of detected hands in image coordinates <!-- --> - Landmarks of detected hands in world coordinates |

### Configurations options

This task has the following configuration options:

| Option Name                     | Description                                                                                                                                                                                                                                                                                                                                                                                                   | Value Range                   | Default Value |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------- |
| `running_mode`                  | Sets the running mode for the task. There are three modes: <br /> IMAGE: The mode for single image inputs. <br /> VIDEO: The mode for decoded frames of a video. <br /> LIVE_STREAM: The mode for a livestream of input data, such as from a camera. In this mode, resultListener must be called to set up a listener to receive results asynchronously.                                                      | {`IMAGE, VIDEO, LIVE_STREAM`} | `IMAGE`       |
| `num_hands`                     | The maximum number of hands detected by the Hand landmark detector.                                                                                                                                                                                                                                                                                                                                           | `Any integer > 0`             | `1`           |
| `min_hand_detection_confidence` | The minimum confidence score for the hand detection to be considered successful in palm detection model.                                                                                                                                                                                                                                                                                                      | `0.0 - 1.0`                   | `0.5`         |
| `min_hand_presence_confidence`  | The minimum confidence score for the hand presence score in the hand landmark detection model. In Video mode and Live stream mode, if the hand presence confidence score from the hand landmark model is below this threshold, Hand Landmarker triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm determines the location of the hand(s) for subsequent landmark detections. | `0.0 - 1.0`                   | `0.5`         |
| `min_tracking_confidence`       | The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. In Video mode and Stream mode of Hand Landmarker, if the tracking fails, Hand Landmarker triggers hand detection. Otherwise, it skips the hand detection.                                                                       | `0.0 - 1.0`                   | `0.5`         |
| `result_callback`               | Sets the result listener to receive the detection results asynchronously when the hand landmarker is in live stream mode. Only applicable when running mode is set to `LIVE_STREAM`                                                                                                                                                                                                                           | N/A                           | N/A           |

## Models

The Hand Landmarker uses a model bundle with two packaged models: a palm detection
model and a hand landmarks detection model. You need a model bundle that
contains both these models to run this task.

> [!WARNING]
> **Attention:** This MediaPipe Solutions Preview is an early release. [Learn more](https://developers.google.com/edge/mediapipe/solutions/about#notice).

| Model name                                                                                                                                   | Input shape          | Quantization type | Model Card                                                                                                                                | Versions                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [HandLandmarker (full)](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) | 192 x 192, 224 x 224 | float 16          | [info](<https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf>) | [Latest](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |

The hand landmark model bundle detects the keypoint localization of 21
hand-knuckle coordinates within the detected hand regions. The model was trained
on approximately 30K real-world images, as well as several rendered synthetic
hand models imposed over various backgrounds.

![](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png)

The hand landmarker model bundle contains a palm detection model and a hand
landmarks detection model. The Palm detection model locates hands within the
input image, and the hand landmarks detection model identifies specific hand
landmarks on the cropped hand image defined by the palm detection model.

Since running the palm detection model is time consuming, when in video or live
stream running mode, Hand Landmarker uses the bounding box defined by the hand
landmarks model in one frame to localize the region of hands for subsequent
frames. Hand Landmarker only re-triggers the palm detection model if the hand
landmarks model no longer identifies the presence of hands or fails to track the
hands within the frame. This reduces the number of times Hand Landmarker tiggers
the palm detection model.

## Task benchmarks

Here's the task benchmarks for the whole pipeline based on the above pre-trained
models. The latency result is the average latency on Pixel 6 using CPU / GPU.

| Model Name            | CPU Latency | GPU Latency |
| --------------------- | ----------- | ----------- |
| HandLandmarker (full) | 17.12ms     | 12.27ms     |

# Web platform guide

The MediaPipe Hand Landmarker task lets you detect the landmarks of the hands in an image.
These instructions show you how to use the Hand Landmarker
for web and JavaScript apps.

For more information about the capabilities, models, and configuration options
of this task, see the [Overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index).

## Code example

The example code for Hand Landmarker provides a complete implementation of this
task in JavaScript for your reference. This code helps you test this task and get
started on building your own hand landmark detection app. You can view, run, and
edit the Hand Landmarker
[example](https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src/tasks/hand-landmarker.ts)
using just your web browser.

## Setup

This section describes key steps for setting up your development environment
specifically to use Hand Landmarker. For general information on
setting up your web and JavaScript development environment, including
platform version requirements, see the
[Setup guide for web](https://developers.google.com/mediapipe/solutions/setup_web).

### JavaScript packages

Hand Landmarker code is available through the MediaPipe `@mediapipe/tasks-vision`
[NPM](https://www.npmjs.com/search?q=@mediapipe) package. You can
find and download these libraries by following the instructions in the platform
[Setup guide](https://developers.google.com/mediapipe/solutions/setup_web#downloads).

You can install the required packages through NPM
using the following command:

    npm install @mediapipe/tasks-vision

If you want to import the task code via a content delivery network (CDN)
service, add the following code in the \<head\> tag in your HTML file:

    <!-- You can replace JSDeliver with another CDN if you prefer to -->
    <head>
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs"
        crossorigin="anonymous"></script>
    </head>

### Model

The MediaPipe Hand Landmarker task requires a trained model that is compatible with this
task. For more information on available trained models for Hand Landmarker, see
the task overview [Models section](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models).

Select and download a model, and then store it within your project directory:

    <dev-project-root>/app/shared/models/

## Create the task

Use one of the Hand Landmarker `createFrom...()` functions to
prepare the task for running inferences. Use the `createFromModelPath()`
function with a relative or absolute path to the trained model file.
If your model is already loaded into memory, you can use the
`createFromModelBuffer()` method.

The code example below demonstrates using the `createFromOptions()` function to
set up the task. The `createFromOptions` function allows you to customize the
Hand Landmarker with configuration options. For more information on configuration
options, see [Configuration options](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js#configuration_options).

The following code demonstrates how to build and configure the task with custom
options:

    const vision = await FilesetResolver.forVisionTasks(
      // path/to/wasm/root
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    const handLandmarker = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: "hand_landmarker.task"
          },
          numHands: 2
        });

### Configuration options

This task has the following configuration options for Web and JavaScript
applications:

| Option Name                  | Description                                                                                                                                                                                                                                                                                                                                                                                                   | Value Range       | Default Value |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------- |
| `runningMode`                | Sets the running mode for the task. There are two modes: <br /> IMAGE: The mode for single image inputs. <br /> VIDEO: The mode for decoded frames of a video or on a livestream of input data, such as from a camera.                                                                                                                                                                                        | {`IMAGE, VIDEO`}  | `IMAGE`       |
| `numHands`                   | The maximum number of hands detected by the Hand landmark detector.                                                                                                                                                                                                                                                                                                                                           | `Any integer > 0` | `1`           |
| `minHandDetectionConfidence` | The minimum confidence score for the hand detection to be considered successful in palm detection model.                                                                                                                                                                                                                                                                                                      | `0.0 - 1.0`       | `0.5`         |
| `minHandPresenceConfidence`  | The minimum confidence score for the hand presence score in the hand landmark detection model. In Video mode and Live stream mode, if the hand presence confidence score from the hand landmark model is below this threshold, Hand Landmarker triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm determines the location of the hand(s) for subsequent landmark detections. | `0.0 - 1.0`       | `0.5`         |
| `minTrackingConfidence`      | The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. In Video mode and Stream mode of Hand Landmarker, if the tracking fails, Hand Landmarker triggers hand detection. Otherwise, it skips the hand detection.                                                                       | `0.0 - 1.0`       | `0.5`         |

## Prepare data

Hand Landmarker can detect hand landmarks in images in any format supported by the
host browser. The task also handles data input preprocessing, including
resizing, rotation and value normalization. To detect hand landmarks in videos,
you can use the API to quickly process one frame at a time, using the timestamp
of the frame to determine when the hand landmarks occur within the video.

## Run the task

The Hand Landmarker uses the `detect()` (with running mode `image`) and
`detectForVideo()` (with running mode `video`) methods to trigger
inferences. The task processes the data, attempts to detect hand landmarks, and
then reports the results.

Calls to the Hand Landmarker `detect()` and `detectForVideo()` methods run
synchronously and block the user interface thread. If you detect hand landmarks
in video frames from a device's camera, each detection blocks the main
thread. You can prevent this by implementing web workers to run the `detect()`
and `detectForVideo()` methods on another thread.

The following code demonstrates how execute the processing with the task model:

### Image

```
const image = document.getElementById("image") as HTMLImageElement;
const handLandmarkerResult = handLandmarker.detect(image);
```

### Video

```
await handLandmarker.setOptions({ runningMode: "video" });

let lastVideoTime = -1;
function renderLoop(): void {
  const video = document.getElementById("video");

  if (video.currentTime !== lastVideoTime) {
    const detections = handLandmarker.detectForVideo(video);
    processResults(detections);
    lastVideoTime = video.currentTime;
  }

  requestAnimationFrame(() => {
    renderLoop();
  });
}
```

For a more complete implementation of running an Hand Landmarker task, see the
[example](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/workers/hand-landmarker.worker.ts).

## Handle and display results

The Hand Landmarker generates a hand landmarker result object for each detection
run. The result object contains hand landmarks in image coordinates, hand
landmarks in world coordinates and handedness(left/right hand) of the detected
hands.

The following shows an example of the output data from this task:

The `HandLandmarkerResult` output contains three components. Each component is an array, where each element contains the following results for a single detected hand:

- Handedness

  Handedness represents whether the detected hands are left or right hands.

- Landmarks

  There are 21 hand landmarks, each composed of `x`, `y` and `z` coordinates. The
  `x` and `y` coordinates are normalized to \[0.0, 1.0\] by the image width and
  height, respectively. The `z` coordinate represents the landmark depth, with
  the depth at the wrist being the origin. The smaller the value, the closer the
  landmark is to the camera. The magnitude of `z` uses roughly the same scale as
  `x`.

- World Landmarks

  The 21 hand landmarks are also presented in world coordinates. Each landmark
  is composed of `x`, `y`, and `z`, representing real-world 3D coordinates in
  meters with the origin at the hand's geometric center.

  HandLandmarkerResult:
  Handedness:
  Categories #0:
  index : 0
  score : 0.98396
  categoryName : Left
  Landmarks:
  Landmark #0:
  x : 0.638852
  y : 0.671197
  z : -3.41E-7
  Landmark #1:
  x : 0.634599
  y : 0.536441
  z : -0.06984
  ... (21 landmarks for a hand)
  WorldLandmarks:
  Landmark #0:
  x : 0.067485
  y : 0.031084
  z : 0.055223
  Landmark #1:
  x : 0.063209
  y : -0.00382
  z : 0.020920
  ... (21 world landmarks for a hand)

The following image shows a visualization of the task output:

![A hand in a thumbs up motion with the skeletal structure of the hand mapped out](https://developers.google.com/static/mediapipe/images/solutions/gesture-recognizer.png)

The Hand Landmarker example code demonstrates how to display the
results returned from the task, see the
[example](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/hand-landmarker.ts)

# Example code snippet

```ts
/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { HandLandmarkerResult, DrawingUtils, HandLandmarker } from "@mediapipe/tasks-vision";
import { BaseVisionTask } from "../components/base-vision-task";

// @ts-ignore
import template from "../templates/hand-landmarker.html?raw";
// @ts-ignore

class HandLandmarkerTask extends BaseVisionTask {
  private drawingUtils: DrawingUtils | undefined;

  private numHands = 2;
  private minHandDetectionConfidence = 0.5;
  private minHandPresenceConfidence = 0.5;
  private minTrackingConfidence = 0.5;

  protected override onInitializeUI() {
    // Confidence Sliders
    const setupSlider = (id: string, onChange: (val: number) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      const valueDisplay = document.getElementById(`${id}-value`)!;
      if (input && valueDisplay) {
        input.addEventListener("input", () => {
          const val = parseFloat(input.value);
          valueDisplay.innerText = val.toString();
          onChange(val);
        });
      }
    };

    setupSlider("min-hand-detection-confidence", (val) => {
      this.minHandDetectionConfidence = val;
      this.worker?.postMessage({ type: "SET_OPTIONS", minHandDetectionConfidence: this.minHandDetectionConfidence });
      this.triggerRedetection();
    });

    setupSlider("min-hand-presence-confidence", (val) => {
      this.minHandPresenceConfidence = val;
      this.worker?.postMessage({ type: "SET_OPTIONS", minHandPresenceConfidence: this.minHandPresenceConfidence });
      this.triggerRedetection();
    });

    setupSlider("min-tracking-confidence", (val) => {
      this.minTrackingConfidence = val;
      this.worker?.postMessage({ type: "SET_OPTIONS", minTrackingConfidence: this.minTrackingConfidence });
      this.triggerRedetection();
    });

    setupSlider("num-hands", (val) => {
      this.numHands = val;
      this.worker?.postMessage({ type: "SET_OPTIONS", numHands: this.numHands });
      this.triggerRedetection();
    });

    // Custom model options for Hand Landmarker
    this.models = {
      hand_landmarker: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([{ label: "Hand Landmarker", value: "hand_landmarker", isDefault: true }]);
    }
  }

  private triggerRedetection() {
    if (this.runningMode === "IMAGE") {
      const testImage = document.getElementById("test-image") as HTMLImageElement;
      if (testImage && testImage.src) {
        this.detectImage(testImage);
      }
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      numHands: this.numHands,
      minHandDetectionConfidence: this.minHandDetectionConfidence,
      minHandPresenceConfidence: this.minHandPresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
    };
  }

  protected override displayImageResult(result: HandLandmarkerResult) {
    const imageCanvas = document.getElementById("image-canvas") as HTMLCanvasElement;
    const testImage = document.getElementById("test-image") as HTMLImageElement;
    const ctx = imageCanvas.getContext("2d")!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.beginPath();
    ctx.rect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(ctx);
      else this.drawingUtils = new DrawingUtils(ctx);

      for (const landmarks of result.landmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  protected override displayVideoResult(result: HandLandmarkerResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    this.canvasCtx.beginPath();
    this.canvasCtx.rect(0, 0, this.canvasElement.width, this.canvasElement.height);
    this.canvasCtx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(this.canvasCtx);
      else this.drawingUtils = new DrawingUtils(this.canvasCtx);

      for (const landmarks of result.landmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  private drawLandmarks(drawingUtils: DrawingUtils, landmarks: any[]) {
    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 5,
    });
    drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
  }
}

// Singleton instance to support modular cleanup
let activeTask: HandLandmarkerTask | null = null;

export async function setupHandLandmarker(container: HTMLElement) {
  activeTask = new HandLandmarkerTask({
    container,
    template,
    defaultModelName: "hand_landmarker",
    defaultModelUrl: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    workerFactory: () => new Worker(new URL("../workers/hand-landmarker.worker.ts", import.meta.url), { type: "module" }),
  });

  await activeTask.initialize();
}

export function cleanupHandLandmarker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
```

```ts filename=base-vision-task.ts
/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { ViewToggle } from "./view-toggle";
import { BaseTask, BaseTaskOptions } from "./base-task";

export interface BaseVisionTaskOptions extends BaseTaskOptions {}

export abstract class BaseVisionTask extends BaseTask {
  protected runningMode: "IMAGE" | "VIDEO" = "IMAGE";
  protected video!: HTMLVideoElement;
  protected canvasElement!: HTMLCanvasElement;
  protected canvasCtx!: CanvasRenderingContext2D;
  protected enableWebcamButton!: HTMLButtonElement;

  protected lastVideoTimeSeconds = -1;
  protected lastTimestampMs = -1;
  protected animationFrameId: number | undefined;

  public override async initialize() {
    this.container.innerHTML = this.options.template;

    this.video = document.getElementById("webcam") as HTMLVideoElement;
    this.canvasElement = document.getElementById("output_canvas") as HTMLCanvasElement;
    if (this.canvasElement) {
      this.canvasCtx = this.canvasElement.getContext("2d")!;
    }
    this.enableWebcamButton = document.getElementById("webcamButton") as HTMLButtonElement;

    this.initWorker();
    this.setupUI();
    this.setupViewToggle();
    this.setupImageUpload();

    // Child class hook
    this.onInitializeUI();
    this.setupDelegateSelect();

    await this.initializeTask();
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    switch (type) {
      case "DETECT_RESULT":
        const { mode, result, inferenceTime } = event.data;
        this.updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
        this.updateInferenceTime(inferenceTime);

        if (mode === "IMAGE") {
          this.displayImageResult(result);
        } else if (mode === "VIDEO") {
          this.displayVideoResult(result);
          if (this.video.srcObject && !this.video.paused) {
            this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
          }
        }
        break;
      default:
        super.handleWorkerMessage(event);
        break;
    }
  }

  protected override handleInitDone() {
    super.handleInitDone();

    if (this.video && this.video.srcObject && this.enableWebcamButton) {
      this.enableWebcamButton.innerText = "Disable Webcam";
      this.enableWebcamButton.disabled = false;
    } else if (this.enableWebcamButton && this.enableWebcamButton.innerText !== "Starting...") {
      this.enableWebcamButton.innerText = "Enable Webcam";
      this.enableWebcamButton.disabled = false;
    }

    if (this.runningMode === "VIDEO") {
      if (this.video.srcObject) {
        this.enableCam();
      }
    } else if (this.runningMode === "IMAGE") {
      const testImage = document.getElementById("test-image") as HTMLImageElement;
      if (testImage && testImage.style.display !== "none" && testImage.src) {
        this.triggerImageDetection(testImage);
      }
    }
  }

  protected setupViewToggle() {
    const viewWebcam = document.getElementById("view-webcam");
    const viewImage = document.getElementById("view-image");

    if (!viewWebcam || !viewImage) return;

    const switchView = (mode: "VIDEO" | "IMAGE") => {
      localStorage.setItem("mediapipe-running-mode", mode);
      const webcamControls = document.getElementById("webcam-controls-container");
      const classificationResults = document.getElementById("classification-results");

      // Clear out old results so they don't linger across mode switches
      if (classificationResults) {
        classificationResults.innerHTML = "";
      }

      if (mode === "VIDEO") {
        viewWebcam.classList.add("active");
        viewImage.classList.remove("active");
        if (webcamControls) webcamControls.style.display = "flex";
        this.runningMode = "VIDEO";
        this.worker?.postMessage({ type: "SET_OPTIONS", runningMode: "VIDEO" });

        const isWebcamActive = localStorage.getItem("mediapipe-webcam-active") === "true";
        if (isWebcamActive) {
          this.enableCam();
        }
      } else {
        viewWebcam.classList.remove("active");
        viewImage.classList.add("active");
        if (webcamControls) webcamControls.style.display = "none";
        this.runningMode = "IMAGE";
        this.worker?.postMessage({ type: "SET_OPTIONS", runningMode: "IMAGE" });
        this.stopCam(false);

        if (this.isWorkerReady) {
          const testImage = document.getElementById("test-image") as HTMLImageElement;
          if (testImage && testImage.src) this.triggerImageDetection(testImage);
        }
      }
    };

    const storedMode = localStorage.getItem("mediapipe-running-mode") as "VIDEO" | "IMAGE";
    const initialMode = storedMode || "IMAGE";

    const viewToggle = new ViewToggle(
      "view-mode-toggle",
      [
        { label: "Webcam", value: "video" },
        { label: "Image", value: "image" },
      ],
      initialMode.toLowerCase(),
      (value) => {
        switchView(value === "video" ? "VIDEO" : "IMAGE");
      },
    );

    viewToggle.setActive(initialMode.toLowerCase());

    switchView(initialMode);
    if (this.enableWebcamButton) {
      this.enableWebcamButton.addEventListener("click", this.toggleCam.bind(this));
    }
  }

  protected setupImageUpload() {
    const imageUpload = document.getElementById("image-upload") as HTMLInputElement;
    const imagePreviewContainer = document.getElementById("image-preview-container")!;
    const testImage = document.getElementById("test-image") as HTMLImageElement;
    const dropzone = document.querySelector(".upload-dropzone") as HTMLElement;
    const dropzoneContent = document.querySelector(".dropzone-content") as HTMLElement;

    if (testImage && testImage.src && dropzoneContent) {
      dropzoneContent.style.display = "none";
    }

    if (dropzone) {
      dropzone.addEventListener("click", (e) => {
        const previewContainer = dropzone.querySelector(".preview-container");
        if (previewContainer && previewContainer.contains(e.target as Node)) {
          return;
        }
        imageUpload?.click();
      });
    }

    imageUpload?.addEventListener("change", (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (testImage) testImage.src = e.target?.result as string;
          if (imagePreviewContainer) imagePreviewContainer.style.display = "";
          const dc = document.querySelector(".dropzone-content") as HTMLElement;
          if (dc) dc.style.display = "none";

          if (testImage) this.triggerImageDetection(testImage);
        };
        reader.readAsDataURL(file);
      }
    });
  }

  protected override async initializeTask() {
    if (this.enableWebcamButton) {
      this.enableWebcamButton.disabled = true;
      if (!this.video || !this.video.srcObject) {
        this.enableWebcamButton.innerText = "Initializing...";
      }
    }
    await super.initializeTask();
  }

  protected override getWorkerInitParamsInner(): Record<string, any> {
    return {
      runningMode: this.runningMode,
      ...this.getWorkerInitParams(),
    };
  }

  protected triggerImageDetection(image: HTMLImageElement) {
    if (image.complete && image.naturalWidth > 0) {
      this.detectImage(image);
    } else {
      image.onload = () => {
        if (image.naturalWidth > 0) {
          this.detectImage(image);
        }
      };
    }
  }

  protected async detectImage(image: HTMLImageElement) {
    if (!this.worker || !this.isWorkerReady) return;
    if (this.runningMode !== "IMAGE") this.runningMode = "IMAGE";

    const bitmap = await createImageBitmap(image);
    this.updateStatus(`Processing image...`);
    this.worker.postMessage(
      {
        type: "DETECT_IMAGE",
        bitmap: bitmap,
        timestampMs: performance.now(),
      },
      [bitmap],
    );
  }

  protected async enableCam() {
    if (!this.worker || !this.video) return;
    if (this.video.srcObject) return;

    if (this.enableWebcamButton) {
      this.enableWebcamButton.innerText = "Starting...";
      this.enableWebcamButton.disabled = true;
    }
    const constraints = { video: true };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (!this.worker || !this.video) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      this.video.srcObject = stream;
      const placeholder = document.getElementById("webcam-placeholder");
      if (placeholder) placeholder.style.display = "none";

      const playAndPredict = () => {
        if (!this.video) return;
        this.video.play().catch(console.error);
        this.predictWebcam();
      };

      if (this.video.readyState >= 2) {
        playAndPredict();
      } else {
        this.video.addEventListener("loadeddata", playAndPredict, { once: true });
      }

      this.runningMode = "VIDEO";
      localStorage.setItem("mediapipe-webcam-active", "true");
      this.worker.postMessage({ type: "SET_OPTIONS", runningMode: "VIDEO" });
      this.updateStatus("Webcam running...");
      if (this.enableWebcamButton) {
        this.enableWebcamButton.innerText = "Disable Webcam";
        this.enableWebcamButton.disabled = false;
      }
    } catch (err) {
      console.error(err);
      this.updateStatus("Camera error!");
      if (this.enableWebcamButton) {
        this.enableWebcamButton.innerText = "Enable Webcam";
        this.enableWebcamButton.disabled = false;
      }
    }
  }

  protected toggleCam() {
    if (this.video && this.video.srcObject) {
      this.stopCam(true);
    } else {
      this.enableCam();
    }
  }

  protected stopCam(persistState = true) {
    if (this.video && this.video.srcObject) {
      const stream = this.video.srcObject as MediaStream;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      this.video.srcObject = null;
      const placeholder = document.getElementById("webcam-placeholder");
      if (placeholder) placeholder.style.display = "flex";
      if (this.enableWebcamButton) this.enableWebcamButton.innerText = "Enable Webcam";
      if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

      if (this.canvasCtx && this.canvasElement) {
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
      }

      if (persistState) {
        localStorage.setItem("mediapipe-webcam-active", "false");
      }
    }
  }

  protected async predictWebcam() {
    if (this.runningMode === "IMAGE") {
      this.runningMode = "VIDEO";
    }

    if (!this.isWorkerReady || !this.worker) {
      this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
      return;
    }

    if (this.video.currentTime !== this.lastVideoTimeSeconds) {
      this.lastVideoTimeSeconds = this.video.currentTime;

      try {
        let bitmap: ImageBitmap;
        if (navigator.webdriver) {
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = this.video.videoWidth || 640;
          tempCanvas.height = this.video.videoHeight || 480;
          const ctx = tempCanvas.getContext("2d", { willReadFrequently: true });
          ctx?.drawImage(this.video, 0, 0, tempCanvas.width, tempCanvas.height);
          bitmap = await window.createImageBitmap(tempCanvas);
        } else {
          bitmap = await window.createImageBitmap(this.video);
        }

        const now = performance.now();
        const timestampMs = now > this.lastTimestampMs ? now : this.lastTimestampMs + 1;
        this.lastTimestampMs = timestampMs;

        this.worker?.postMessage(
          {
            type: "DETECT_VIDEO",
            bitmap: bitmap,
            timestampMs: timestampMs,
          },
          [bitmap],
        );
      } catch (e) {
        console.error("Failed to create ImageBitmap from video", e);
        this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
      }
    } else {
      this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
    }
  }

  public override cleanup() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    this.stopCam(false);

    if (this.canvasCtx && this.canvasElement) {
      this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }

    super.cleanup();
  }

  protected abstract displayImageResult(result: any): void;
  protected abstract displayVideoResult(result: any): void;
}
```
