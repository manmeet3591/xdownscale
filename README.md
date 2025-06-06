# xdownscale
---

````markdown

xdownscale is a Python package for super-resolution downscaling of satellite data using different AI models. It allows mapping from a lower-resolution input image (e.g., VIIRS) to a higher-resolution target (e.g., DMSP) in just two lines of code using PyTorch and xarray.

Installation

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

## Requirements

* `torch`
* `xarray`
* `numpy`

These are automatically installed via `setup.py`.

## Usage

```python
import xarray as xr
import numpy as np
from xdownscale import Downscaler

# Create dummy data for test
x = np.random.rand(128, 128).astype(np.float32)
y = (x + np.random.normal(0, 0.01, size=x.shape)).astype(np.float32)

input_da = xr.DataArray(x, dims=["lat", "lon"])
target_da = xr.DataArray(y, dims=["lat", "long"])

ds = Downscaler(input_da, target_da, model_name="fsrcnn") # use other models by changing the model name here

# Predict on new input
result = ds.predict(input_da)
result.plot()

```

## Description

* Training is performed patch-wise using a simple DataLoader in PyTorch.
* The `predict` method returns an `xarray.DataArray` with the same dimensions as the input.

## Development

To extend this package:

* Edit the model architecture in `xdownscale/model.py`
* Add training logic to `xdownscale/core.py`
* Customize patching utilities in `xdownscale/utils.py`

## License

MIT License.
