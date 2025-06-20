# xdownscale
<p align="center"><img src="https://raw.githubusercontent.com/manmeet3591/xdownscale/main/xdownscale.png" alt="xdownscale logo" width="400"/></p>
<p align="center">
  <a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active"></a>
  <a href="https://pypi.org/project/xdownscale/"><img src="https://badge.fury.io/py/xdownscale.svg" alt="PyPI version"></a>
  <!-- <a href="https://anaconda.org/conda-forge/solweig-gpu"><img src="https://anaconda.org/conda-forge/solweig-gpu/badges/version.svg" alt="Conda version"></a> -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pepy.tech/project/xdownscale"><img src="https://pepy.tech/badge/xdownscale" alt="PyPI Downloads"></a>
  <a href="https://joss.theoj.org/papers/40c9c6c313ca7fcd1a236955c391d54c"><img src="https://joss.theoj.org/papers/40c9c6c313ca7fcd1a236955c391d54c/status.svg"></a>
</p>

xdownscale is a Python package for super-resolution downscaling of gridded datasets using deep learning. It supports a wide range of applications, including satellite observations, reanalysis data, and climate model outputs. Built with PyTorch and xarray, it enables efficient mapping from coarse-to-fine-resolution grids in just a few lines of code. 

---

## Installation

To install from PyPI, we recommend using a conda environment

```bash
conda create -n xdownscale python=3.10
conda activate xdownscale
conda install -c conda-forge pytorch cudatoolkit=11.8 cudnn
pip install xdownscale
```

To install from source:

```bash
git clone https://github.com/manmeet3591/xdownscale.git
cd xdownscale
pip install .
```

Or install from a zipped archive:

```bash
unzip xdownscale_package.zip
cd xdownscale
pip install .
```

---

## Usage

```python
import xarray as xr
import numpy as np
from xdownscale import Downscaler

# Create dummy coarse-resolution input and fine-resolution target
x = np.random.rand(128, 128).astype(np.float32)
y = (x + np.random.normal(0, 0.01, size=x.shape)).astype(np.float32)

input_da = xr.DataArray(x, dims=["lat", "lon"])
target_da = xr.DataArray(y, dims=["lat", "long"])

# Initialize the downscaler
ds = Downscaler(input_da, target_da, model_name="fsrcnn")

# Predict high-resolution output
result = ds.predict(input_da)
result.plot()
```

**Available models**:  
`srcnn`, `fsrcnn`, `lapsr`, `carnm`, `falsra`, `falsrb`, `srresnet`, `carn`, `oisrrk2`, `mdsr`, `san`, `rcan`, `unet`, `dlgsanet`, `dpmn`, `safmn`, `dpt`, `distgssr`, `swin`

---

## Description

xdownscale performs patch-wise training using PyTorch’s `DataLoader` and returns predictions as `xarray.DataArray` objects. It is designed to work with any gridded dataset and provides a flexible interface for model selection, training, and inference.

---

## Sample Data

Sample input and target data are provided in the `data/` directory for testing and demonstrations.

---

## Development

To extend or customize the package:

- Modify model architectures in `xdownscale/model.py`
- Add training logic in `xdownscale/core.py`
- Customize patch extraction and utilities in `xdownscale/utils.py`

---

## License

This project is licensed under the MIT License.
