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

### Step 3: Run the Application

Launch the script directly with `uv`:

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
uv run python main.py
```

#### First Run Compile

yolo26s-seg.onnx: 10 minutes
yolo26n-seg.onnx: 5 minutes

CPU will spike, then GPU will spike
