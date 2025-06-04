import torch
import numpy as np
from .model import * # SRCNN, FSRCNN
from .utils import patchify, unpatchify
import xarray as xr

class Downscaler:
    def __init__(self, input_da, target_da, patch_size=32, device='cuda', model_name="srcnn", epochs=100,):
        self.patch_size = patch_size
        self.device = device
        self.x_max = input_da.values.max()
        self.y_max = target_da.values.max()
        self.input_da = input_da / self.x_max
        self.target_da = target_da / self.y_max
        self.epochs = epochs

        self.model = self._get_model(model_name).to(device)
        self._train()
    
    def _get_model(self, name):
        name = name.lower()
        if name == "srcnn":
            return SRCNN()
        elif name == "fsrcnn":
            return FSRCNN()
        else:
            raise ValueError(f"Unknown model name: {name}")

    def _train(self):
        x_train = self.input_da.values.astype(np.float32)
        y_train = self.target_da.values.astype(np.float32)

        x_train_patches = patchify(x_train, self.patch_size)
        y_train_patches = patchify(y_train, self.patch_size)

        x_tensor = torch.from_numpy(x_train_patches[:, None, :, :]).to(self.device)
        y_tensor = torch.from_numpy(y_train_patches[:, None, :, :]).to(self.device)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"[{epoch}] loss: {total_loss/len(loader):.4f}")

    def predict(self, input_da: xr.DataArray) -> xr.DataArray:
        x_input = (input_da.values / self.x_max).astype(np.float32)
        patches = patchify(x_input, self.patch_size)
        x_tensor = torch.from_numpy(patches[:, None, :, :]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(x_tensor).cpu().numpy()[:, 0, :, :] * self.y_max

        preds[preds < 0] = 0.0
        reconstructed = unpatchify(preds, x_input.shape)

        return xr.DataArray(reconstructed, coords=input_da.coords, dims=input_da.dims)
