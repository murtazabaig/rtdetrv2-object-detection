# RT-DETRv2 Object Detection (Portfolio Project)

This repository provides a complete, local runnable RT-DETRv2 object detection setup with a Windows CPU demo, model configs, and training/inference utilities.

## Highlights
- PyTorch RT-DETRv2 implementation (primary)
- CPU inference quick start on Windows
- Model configs and scripts for training, testing, and export
- Benchmark and deployment references

## Project Structure
- `rtdetrv2_pytorch/` main PyTorch RT-DETRv2 code, configs, tools, and inference scripts
- `rtdetr_pytorch/` legacy PyTorch RT-DETR (v1)
- `rtdetrv2_paddle/` PaddlePaddle RT-DETRv2 implementation
- `rtdetr_paddle/` legacy PaddlePaddle RT-DETR
- `benchmark/` benchmarking utilities
- `docker/` Docker build and deployment scripts
- `supervisely_integration/` integration examples

## Requirements
- Python 3.11 recommended
- Torch and torchvision (CPU or CUDA build)
- Additional Python packages:
  - `PyYAML`
  - `tensorboard`
  - `scipy`
  - `pycocotools` (required for training/evaluation)

Note: On Windows, inference can run without `pycocotools` due to a lazy import path. Training/evaluation still requires it.

## Setup (Windows)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r rtdetrv2_pytorch\requirements.txt
.\.venv\Scripts\python -m pip install PyYAML tensorboard scipy
```

If you have a CUDA GPU, install the matching CUDA build of torch/torchvision instead of the CPU index URL.

## Quick Demo (CPU)
This runs inference on a bundled sample image and writes the result to `rtdetrv2_pytorch/results_0.jpg`.

```powershell
cd rtdetrv2_pytorch
$env:PYTHONPATH = "$pwd"
..\.venv\Scripts\python references\deploy\rtdetrv2_torch.py `
  -c configs\rtdetrv2\rtdetrv2_r18vd_120e_coco.yml `
  -r rtdetrv2_r18vd_120e_coco_rerun_48.1.pth `
  -f ..\rtdetr_pytorch\image_02.jpg `
  -d cpu
```

### Download the pretrained weights
Place the weight file in `rtdetrv2_pytorch/`:

```powershell
cd rtdetrv2_pytorch
Invoke-WebRequest -Uri "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" -OutFile "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
```

## Inference with Your Own Image
```powershell
cd rtdetrv2_pytorch
$env:PYTHONPATH = "$pwd"
..\.venv\Scripts\python references\deploy\rtdetrv2_torch.py `
  -c configs\rtdetrv2\rtdetrv2_r18vd_120e_coco.yml `
  -r rtdetrv2_r18vd_120e_coco_rerun_48.1.pth `
  -f C:\path\to\your\image.jpg `
  -d cpu
```

## Training (Multi-GPU Example)
```powershell
cd rtdetrv2_pytorch
$env:PYTHONPATH = "$pwd"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools\train.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_6x_coco.yml --use-amp --seed=0
```

## Testing
```powershell
cd rtdetrv2_pytorch
$env:PYTHONPATH = "$pwd"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools\train.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_6x_coco.yml -r path\to\checkpoint.pth --test-only
```

## Export ONNX
```powershell
cd rtdetrv2_pytorch
$env:PYTHONPATH = "$pwd"
..\.venv\Scripts\python tools\export_onnx.py -c configs\rtdetrv2\rtdetrv2_r18vd_120e_coco.yml -r path\to\checkpoint.pth --check
```

## Troubleshooting
- If pip install fails with file-in-use errors, close any running Python processes and retry.
- For training/evaluation on Windows, ensure `pycocotools` is installed. If it is not available on your environment, consider WSL or a Linux setup.

## License
MIT License. See `LICENSE`.
