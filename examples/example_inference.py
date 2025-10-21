"""
Example: Training and Inference with xdownscale
================================================

This example shows how to:
1. Train a downscaling model
2. Perform inference on new data
3. Handle 3D data with time, lat, lon dimensions
4. Save and load models
"""

import numpy as np
import xarray as xr
import torch
from xdownscale import Downscaler

# ============================================================================
# OPTION 1: Basic Inference (immediately after training)
# ============================================================================

def example_basic_inference():
    """Train and immediately run inference"""
    print("=" * 60)
    print("OPTION 1: Basic Inference After Training")
    print("=" * 60)
    
    # Create sample data (low-res and high-res)
    # In practice, you'd load your actual data here
    low_res = xr.DataArray(
        np.random.rand(32, 32),
        dims=['lat', 'lon'],
        coords={'lat': np.linspace(30, 40, 32), 
                'lon': np.linspace(-100, -90, 32)}
    )
    
    high_res = xr.DataArray(
        np.random.rand(32, 32),
        dims=['lat', 'lon'],
        coords={'lat': np.linspace(30, 40, 32),
                'lon': np.linspace(-100, -90, 32)}
    )
    
    # Train the model
    print("\n1. Training the model...")
    model = Downscaler(
        low_res, 
        high_res,
        model_name="srcnn",
        patch_size=16,
        batch_size=8,
        epochs=10,
        device='cpu'  # Use 'cuda' if GPU available
    )
    
    # Inference on the same data
    print("\n2. Running inference...")
    predictions = model.predict(low_res, use_patches=True)
    
    print(f"\nInput shape: {low_res.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output coords preserved: {list(predictions.coords.keys())}")
    print("✓ Inference complete!")


# ============================================================================
# OPTION 2: Inference on New Data
# ============================================================================

def example_new_data_inference():
    """Train on one dataset, predict on another"""
    print("\n" + "=" * 60)
    print("OPTION 2: Inference on New Data")
    print("=" * 60)
    
    # Training data
    train_low = xr.DataArray(
        np.random.rand(64, 64),
        dims=['lat', 'lon']
    )
    train_high = xr.DataArray(
        np.random.rand(64, 64),
        dims=['lat', 'lon']
    )
    
    # Train model
    print("\n1. Training on training data...")
    model = Downscaler(
        train_low, train_high,
        model_name="srcnn",
        epochs=5,
        device='cpu'
    )
    
    # NEW data for inference (different time/location)
    new_data = xr.DataArray(
        np.random.rand(64, 64),
        dims=['lat', 'lon'],
        coords={'lat': np.linspace(40, 50, 64),
                'lon': np.linspace(-110, -100, 64)}
    )
    
    print("\n2. Running inference on NEW data...")
    new_predictions = model.predict(new_data)
    
    print(f"\nNew data shape: {new_data.shape}")
    print(f"Predictions shape: {new_predictions.shape}")
    print("✓ Inference on new data complete!")


# ============================================================================
# OPTION 3: 3D Data with Time Dimension
# ============================================================================

def example_3d_inference():
    """Handle 3D data with time, lat, lon dimensions"""
    print("\n" + "=" * 60)
    print("OPTION 3: 3D Data with Time Dimension")
    print("=" * 60)
    
    # Create 3D data (time, lat, lon)
    time_steps = 10
    lat_size = 32
    lon_size = 32
    
    low_res_3d = xr.DataArray(
        np.random.rand(time_steps, lat_size, lon_size),
        dims=['time', 'lat', 'lon'],
        coords={
            'time': range(time_steps),
            'lat': np.linspace(30, 40, lat_size),
            'lon': np.linspace(-100, -90, lon_size)
        }
    )
    
    high_res_3d = xr.DataArray(
        np.random.rand(time_steps, lat_size, lon_size),
        dims=['time', 'lat', 'lon'],
        coords=low_res_3d.coords
    )
    
    # Train (treats each time step as a separate sample)
    print(f"\n1. Training on 3D data: {low_res_3d.shape}")
    model = Downscaler(
        low_res_3d, high_res_3d,
        model_name="srcnn",
        patch_size=16,
        epochs=5,
        device='cpu'
    )
    
    # Inference on new time steps
    new_time_data = xr.DataArray(
        np.random.rand(5, lat_size, lon_size),  # 5 new time steps
        dims=['time', 'lat', 'lon'],
        coords={
            'time': range(10, 15),  # Different time indices
            'lat': low_res_3d.coords['lat'],
            'lon': low_res_3d.coords['lon']
        }
    )
    
    print(f"\n2. Running inference on new time steps: {new_time_data.shape}")
    predictions_3d = model.predict(new_time_data)
    
    print(f"\nPredictions shape: {predictions_3d.shape}")
    print(f"Dimensions preserved: {predictions_3d.dims}")
    print("✓ 3D inference complete!")


# ============================================================================
# OPTION 4: Save and Load Model for Later Use
# ============================================================================

def example_save_load_model():
    """Save trained model and load it later for inference"""
    print("\n" + "=" * 60)
    print("OPTION 4: Save and Load Model")
    print("=" * 60)
    
    # Train model
    train_data = xr.DataArray(np.random.rand(32, 32), dims=['lat', 'lon'])
    target_data = xr.DataArray(np.random.rand(32, 32), dims=['lat', 'lon'])
    
    print("\n1. Training model...")
    model = Downscaler(
        train_data, target_data,
        model_name="srcnn",
        epochs=5,
        device='cpu'
    )
    
    # Save the model
    print("\n2. Saving model...")
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'x_max': model.x_max,
        'y_max': model.y_max,
        'model_name': 'srcnn',
        'patch_size': model.patch_size
    }, 'trained_model.pth')
    print("✓ Model saved to 'trained_model.pth'")
    
    # Later: Load the model for inference
    print("\n3. Loading model for inference...")
    checkpoint = torch.load('trained_model.pth', map_location='cpu')
    
    # Recreate the model (need to create a dummy Downscaler to get the architecture)
    # Then load the saved weights
    from xdownscale.model import SRCNN
    loaded_model = SRCNN()
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    # Now you can use it for inference manually
    new_data = xr.DataArray(np.random.rand(32, 32), dims=['lat', 'lon'])
    
    print("\n4. Running inference with loaded model...")
    with torch.no_grad():
        x_normalized = new_data.values / checkpoint['x_max']
        x_tensor = torch.from_numpy(x_normalized[None, None, :, :].astype(np.float32))
        prediction = loaded_model(x_tensor).numpy()[0, 0] * checkpoint['y_max']
    
    print(f"Prediction shape: {prediction.shape}")
    print("✓ Inference with loaded model complete!")


# ============================================================================
# OPTION 5: Patch vs Whole Image Inference
# ============================================================================

def example_inference_modes():
    """Compare patch-based vs whole image inference"""
    print("\n" + "=" * 60)
    print("OPTION 5: Patch vs Whole Image Inference")
    print("=" * 60)
    
    # Train model
    train_data = xr.DataArray(np.random.rand(64, 64), dims=['lat', 'lon'])
    target_data = xr.DataArray(np.random.rand(64, 64), dims=['lat', 'lon'])
    
    model = Downscaler(
        train_data, target_data,
        model_name="srcnn",
        patch_size=32,
        epochs=5,
        device='cpu'
    )
    
    test_data = xr.DataArray(np.random.rand(64, 64), dims=['lat', 'lon'])
    
    # Method 1: Patch-based inference (default, recommended)
    print("\n1. Patch-based inference (handles large images)...")
    pred_patches = model.predict(test_data, use_patches=True)
    print(f"   Output shape: {pred_patches.shape}")
    
    # Method 2: Whole image inference (for small images only)
    print("\n2. Whole image inference (faster for small images)...")
    pred_whole = model.predict(test_data, use_patches=False)
    print(f"   Output shape: {pred_whole.shape}")
    
    print("\n✓ Both methods complete!")
    print("\nRecommendation:")
    print("  - Use use_patches=True for large images or limited memory")
    print("  - Use use_patches=False for small images when speed matters")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("XDOWNSCALE INFERENCE EXAMPLES")
    print("=" * 60)
    
    # Run all examples
    example_basic_inference()
    example_new_data_inference()
    example_3d_inference()
    example_save_load_model()
    example_inference_modes()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Use model.predict(data) for inference after training")
    print("2. The model automatically handles 3D data (time, lat, lon)")
    print("3. Coordinates and dimensions are preserved in output")
    print("4. You can save/load models with torch.save/torch.load")
    print("5. Choose use_patches based on your data size")

