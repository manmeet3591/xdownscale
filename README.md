# xdownscale



---

````markdown


**xdownscale** is a Python package for super-resolution downscaling of satellite data using a simple SRCNN (Super-Resolution Convolutional Neural Network) model. It allows mapping from a lower-resolution input image (e.g., VIIRS) to a higher-resolution target (e.g., DMSP) in just two lines of code using PyTorch and xarray.

🚀 Installation

To install from source:

```bash
git clone https://github.com/yourusername/xdownscale.git
cd xdownscale
pip install .
````

Or, if you're using the zip:

```bash
unzip xdownscale_package.zip
cd xdownscale
pip install .
```

## 📦 Requirements

* `torch`
* `xarray`
* `numpy`

These are automatically installed via `setup.py`.

## 🔧 Usage

```python
import xarray as xr
from xdownscale import Downscaler

# Load your input and target data
viirs = xr.open_dataset("viirs.nc")["avg_rad"]
dmsp = xr.open_dataset("dmsp.nc")["constant"]

# Initialize and train the downscaling model
ds = Downscaler(viirs, dmsp)

# Predict high-resolution output from a new VIIRS image
predicted = ds.predict(viirs)
```

## 📘 Description

* This tool uses a pre-defined SRCNN with three convolutional layers.
* Training is performed patch-wise using a simple DataLoader in PyTorch.
* The `predict` method returns an `xarray.DataArray` with the same dimensions as the input.

## 🛠️ Development

To extend this package:

* Edit the model architecture in `xdownscale/model.py`
* Add training logic to `xdownscale/core.py`
* Customize patching utilities in `xdownscale/utils.py`

## 📄 License

MIT License.
