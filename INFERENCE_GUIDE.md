# Quick Inference Guide for xdownscale

## Basic Workflow

### 1. **Train the Model**
```python
import xarray as xr
from xdownscale import Downscaler

# Load your data
low_res_data = xr.open_dataarray('low_res.nc')   # e.g., (time, lat, lon)
high_res_data = xr.open_dataarray('high_res.nc') # e.g., (time, lat, lon)

# Train
model = Downscaler(
    low_res_data,
    high_res_data,
    model_name="srcnn",      # Choose: srcnn, fsrcnn, unet, etc.
    patch_size=32,           # Adjust based on your data
    batch_size=16,
    epochs=100,
    device='cuda',           # Use 'cuda' for GPU, 'cpu' for CPU
    patience=10              # Early stopping patience
)
```

### 2. **Run Inference** ✨

```python
# Inference on new data
new_low_res = xr.open_dataarray('new_data.nc')
predictions = model.predict(new_low_res, use_patches=True)

# Save predictions
predictions.to_netcdf('predictions.nc')
```

---

## Advanced Options

### 3D Data (time, lat, lon)
```python
# Works automatically! Each time step is processed independently
data_3d = xr.DataArray(
    data,  # shape: (100, 64, 64) for 100 time steps
    dims=['time', 'lat', 'lon']
)

model = Downscaler(train_low_3d, train_high_3d, model_name="srcnn")
predictions_3d = model.predict(test_low_3d)  # Returns same dimensions
```

### Inference Modes
```python
# Patch-based (recommended for large images)
predictions = model.predict(data, use_patches=True)

# Whole image (faster for small images < 128x128)
predictions = model.predict(data, use_patches=False)
```

### Save and Load Models
```python
import torch

# After training, save the model
torch.save({
    'model_state_dict': model.model.state_dict(),
    'x_max': model.x_max,
    'y_max': model.y_max,
    'model_name': 'srcnn',
    'patch_size': model.patch_size
}, 'my_model.pth')

# Later, load for inference
from xdownscale.model import SRCNN

checkpoint = torch.load('my_model.pth')
loaded_model = SRCNN()
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# Manual inference
import numpy as np
with torch.no_grad():
    x = new_data.values / checkpoint['x_max']
    x_tensor = torch.from_numpy(x[None, None, :, :].astype(np.float32))
    pred = loaded_model(x_tensor).numpy()[0, 0] * checkpoint['y_max']
```

---

## Complete Example

```python
import xarray as xr
import numpy as np
from xdownscale import Downscaler

# 1. Load your training data
train_input = xr.open_dataarray('dmsp_2012_austin.nc')
train_target = xr.open_dataarray('viirs_2012_austin.nc')

# 2. Train the model
print("Training model...")
model = Downscaler(
    train_input,
    train_target,
    model_name="srcnn",
    patch_size=32,
    batch_size=20,
    epochs=100,
    device='cuda',
    val_split=0.1,
    test_split=0.1,
    patience=10
)

# 3. Load new data for prediction
print("Loading new data...")
new_input = xr.open_dataarray('new_low_res_data.nc')

# 4. Make predictions
print("Making predictions...")
predictions = model.predict(new_input, use_patches=True)

# 5. Save results
print("Saving predictions...")
predictions.to_netcdf('downscaled_predictions.nc')

print("✓ Done!")
```

---

## Model Options

Available models (via `model_name` parameter):
- `"srcnn"` - Fast, simple (recommended for starting)
- `"fsrcnn"` - Faster variant of SRCNN
- `"unet"` - U-Net architecture
- `"srresnet"` - Deep residual network
- `"swin"` - Swin Transformer (state-of-the-art)
- `"rcan"` - Residual Channel Attention Network
- And many more! See `core.py` for full list

---

## Tips

1. **Memory Issues?**
   - Reduce `patch_size` (e.g., 16 instead of 32)
   - Reduce `batch_size` (e.g., 8 instead of 20)
   - Use `use_patches=True` during inference

2. **Slow Training?**
   - Use GPU: `device='cuda'`
   - Reduce `epochs` or use `patience` for early stopping
   - Try faster models: `"fsrcnn"` or `"srcnn"`

3. **Poor Results?**
   - Train longer (increase `epochs`)
   - Use more complex model: `"unet"`, `"srresnet"`, or `"swin"`
   - Ensure input and target data are properly aligned
   - Normalize your data if needed

4. **3D Data?**
   - Just pass it in! The model handles time dimension automatically
   - Each time step is treated as an independent training sample

---

## Questions?

Check the example file: `example_inference.py`
Or see the documentation in the code docstrings.

