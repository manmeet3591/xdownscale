import torch
import numpy as np
from .model import *  # SRCNN, FSRCNN
from .utils import patchify, unpatchify
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

class Downscaler:
    def __init__(self, input_da, target_da, model_name="srcnn",
                 patch_size=32, batch_size=20, epochs=100,
                 val_split=0.1, test_split=0.1, device='cuda',
                 use_wandb=False):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.use_wandb = use_wandb

        self.x_max = input_da.values.max()
        self.y_max = target_da.values.max()
        self.input_da = input_da / self.x_max
        self.target_da = target_da / self.y_max

        self.model = self._get_model(model_name).to(device)
        self._train(val_split, test_split, model_name)

    def _get_model(self, name):
        name = name.lower()
        if name == "srcnn":
            return SRCNN()
        elif name == "fsrcnn":
            return FSRCNN()
        elif name == "lapsr":
            return LapSRN(in_channels=1, upscale_factor=1)
        elif name == "carnm":
            return CARNM(num_channels=1, scale_factor=1)
        elif name == "falsrb":
            return FALSRB(in_channels=1, out_channels=1, scale_factor=1)
        elif name == "srresnet":
            return SRResNet(in_channels=1, out_channels=1, upscale_factor=1)
        elif name == "carn":
            return CARN(in_channels=1, out_channels=1, upscale_factor=1)
        elif name == "falsra":
            return FALSR_A()
        elif name == "oisrrk2":
            return OISRRK2()
        elif name == "mdsr":
            return MDSR(in_channels=1, upscale_factor=1, num_blocks=16)
        elif name == "san":
            return SAN(in_channels=1, upscale_factor=1, num_blocks=16, num_heads=8)
        elif name == "rcan":
            return RCAN(in_channels=1, num_blocks=1, upscale_factor=16)
        elif name == "unet":
            return UNet(in_channels=1, out_channels=1)
        else:
            raise ValueError(f"Unknown model name: {name}")

    def _train(self, val_split, test_split, model_name):
        if self.use_wandb:
            wandb.init(project="xdownscale", config={
                "model": model_name,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patch_size": self.patch_size
            })
            wandb.init(
                project="xdownscale",
                name=f"{model_name.upper()}_run",
                config={
                    "model": model_name,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "patch_size": self.patch_size
                }
            )

        x = self.input_da.values.astype(np.float32)
        y = self.target_da.values.astype(np.float32)

        x_patches = patchify(x, self.patch_size)
        y_patches = patchify(y, self.patch_size)

        x_tensor = torch.from_numpy(x_patches[:, None, :, :])
        y_tensor = torch.from_numpy(y_patches[:, None, :, :])

        dataset = TensorDataset(x_tensor, y_tensor)
        total = len(dataset)
        val_len = int(total * val_split)
        test_len = int(total * test_split)
        train_len = total - val_len - test_len

        train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.model(xb)
                    val_loss += criterion(preds, yb).item()
            val_loss /= len(val_loader)

            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"[{epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        self.test_loader = test_loader
        if self.use_wandb:
            wandb.finish()

    def predict(self, input_da: xr.DataArray, use_patches: bool = True) -> xr.DataArray:
        x_input = (input_da.values / self.x_max).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            if use_patches:
                patches = patchify(x_input, self.patch_size)
                x_tensor = torch.from_numpy(patches[:, None, :, :]).to(self.device)
                preds = self.model(x_tensor).cpu().numpy()[:, 0, :, :] * self.y_max
                preds[preds < 0] = 0.0
                reconstructed = unpatchify(preds, x_input.shape)
            else:
                x_tensor = torch.from_numpy(x_input[None, None, :, :]).to(self.device)
                pred = self.model(x_tensor).cpu().numpy()[0, 0, :, :] * self.y_max
                pred[pred < 0] = 0.0
                reconstructed = pred

        return xr.DataArray(reconstructed, coords=input_da.coords, dims=input_da.dims)
