# Installation

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
- GCC 5.4+

## Prepare the Environment

1. Use conda and activate the environment:

   ```bash
   conda create -n onedl-mm python=3.10 -y
   conda activate onedl-mm
   ```

2. Install PyTorch

   Before installing `MMEngine`, please make sure that PyTorch has been successfully installed in the environment. You can refer to [PyTorch official installation documentation](https://pytorch.org/get-started/locally/#start-locally). Verify the installation with the following command:

   ```bash
   python -c 'import torch;print(torch.__version__)'
   ```

## Install MMEngine

### Install with mim (recommended)

[mim](https://github.com/vbti-development/mim) is a package management tool for OpenMMLab projects, which can be used to install the OpenMMLab project easily.

```bash
pip install -U onedl-mim
mim install onedl-mmengine
```

### Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv pip install onedl-mmengine
```

### Install with pip

```bash
pip install onedl-mmengine
```

### Use docker images

1. Build the image

   ```bash
   docker build -t mmengine https://github.com/vbti-development/onedl-mmengine.git#main:docker/release
   ```

   More information can be referred from [onedl-mmengine/docker](https://github.com/vbti-development/onedl-mmengine/tree/main/docker).

2. Run the image

   ```bash
   docker run --gpus all --shm-size=8g -it onedl-mmengine
   ```

### Build from source

#### Build mmengine

```bash
# if cloning speed is too slow, you can switch the source to https://gitee.com/vbti-development/onedl-mmengine.git
git clone https://github.com/vbti-development/onedl-mmengine.git
cd onedl-mmengine
pip install -e . -v
```

## Verify the Installation

To verify if `MMEngine` and the necessary environment are successfully installed, we can run this command:

```bash
python -c 'import mmengine;print(mmengine.__version__)'
```
