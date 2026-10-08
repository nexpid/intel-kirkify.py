# kirkify.py

> [!NOTE]
> A vibecoded fork of kirkify.py, with added Intel iGPU support

Groundbreaking engine behind the hit website, kirkify.me, which some are saying is the most innovative software since Facebook or Google.

It's a relatively simple python script to kirkify any image or video given.

## Setup

It's a bit of a pain to setup but I promise it's worth it.

First download the face swapping AI (528mb)

```bash
curl -o "inswapper_128.onnx" https://bk4vz20t6s.ufs.sh/f/5eVwDsd8R3jL5kumGF8R3jLVwUJfdOu8cQ4ymMqAFeW7zrEX
```

Next, install the requirements.

```bash
pip install -r requirements.txt
```

After, initialize the script (you don't technically have to but it saves time on first run)

```bash
python3 kirkify.py init
```

Finally, if you don't already have it installed, the video kirkifier requires `ffmpeg` to be installed

```bash
apt install ffmpeg
```

### GPU Acceleration (Optional)

To run face detection and swapping on the GPU, you need ONNX Runtime with CUDA support and a compatible NVIDIA driver:

- NVIDIA GPU with recent driver (CUDA 12 compatible recommended)
- Install ONNX Runtime GPU wheels via pip

Basic installation:

```bash
pip install onnxruntime-gpu
```

If you still see missing CUDA DLLs, install the CUDA dependency wheels explicitly (CUDA 12 series):

```bash
pip install onnxruntime-gpu nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
```

How to verify it's working:

```bash
python kirkify.py input.mp4 output.mp4 --gpu --fast
```

You should see a line like: `INSwapper active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']`.
If CUDA is not available, the script will fall back to CPU and print a warning.

### Intel Integrated GPU Acceleration (Optional)

On Linux, Intel integrated GPU acceleration uses ONNX Runtime's OpenVINO execution provider. Install the OpenVINO build in place of the default `onnxruntime` package:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-openvino
```

Make sure the Intel GPU driver and Level Zero runtime are installed, then run:

```bash
python kirkify.py input.mp4 output.mp4 --intel --fast
```

The script uses OpenVINO's GPU device and falls back to CPU if the provider is unavailable. You should see `OpenVINOExecutionProvider` in the reported active providers. Do not install `onnxruntime` and `onnxruntime-openvino` together in the same environment; use the CUDA instructions above for NVIDIA GPUs instead.

## Usage

```bash
python3 kirkify.py <input_media> [output_path]
```

### Performance and Device Flags

The script supports several flags to speed up processing and to select the compute device. These are optional and can be combined.

- `--fast`: Use a smaller face detector size (320x320) to speed up detection (slight quality trade-off).
- `--frame-step N`: Process 1 out of every N frames (video only). Frames are renumbered to keep the output sequence contiguous.
- `--workers M`: Use M threads to process frames in parallel (video only). A good default is the number of CPU cores.
- `--gpu`: Prefer GPU for face detection and INSwapper when available.
- `--intel`: Use an Intel integrated GPU through OpenVINO when available.
- `--cpu`: Force CPU even if a GPU is present.

Examples (cmd/PowerShell):

```bash
# Fast mode using GPU if available
python kirkify.py input.mp4 output.mp4 --fast --gpu

# Skip frames to speed up and parallelize on 8 threads
python kirkify.py input.mp4 output.mp4 --fast --frame-step 3 --workers 8 --gpu

# Force CPU and keep full quality detector
python kirkify.py input.mp4 output.mp4 --cpu
```

Notes:
- GPU mode requires ONNX Runtime GPU (onnxruntime-gpu) installed and a compatible NVIDIA driver. If `--gpu` is requested but CUDA is not available, the script will fall back to CPU and print a warning.
- Intel mode requires `onnxruntime-openvino` and a working Intel GPU driver/runtime. If `--intel` is requested but OpenVINO is not available, the script will fall back to CPU and print a warning.
- When using `--frame-step`, only the selected frames are processed; the output video remains valid because frames are renumbered sequentially.
- `ffmpeg` must be installed and available in your PATH.
