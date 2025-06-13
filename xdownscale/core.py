import torch
import numpy as np
from .model import *  # SRCNN, FSRCNN
from .distgssr import Net as distgssr
from .swinir import SwinIR
from .utils import patchify, unpatchify
from .srresnet import SRResNet
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
import torch.nn as nn
import torch.nn.functional as F
import copy

class Downscaler:
    def __init__(self, input_da, target_da, model_name="srcnn",
                 patch_size=32, batch_size=20, epochs=100,
                 val_split=0.1, test_split=0.1, device='cuda',
                 use_wandb=False, patience=10, min_delta=1e-4):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.use_wandb = use_wandb
        self.patience = patience
        self.min_delta = min_delta

        self.x_max = input_da.values.max()
        self.y_max = target_da.values.max()
        self.input_da = input_da / self.x_max
        self.target_da = target_da / self.y_max

        self.model = self._get_model(model_name).to(device)
        self._train(val_split, test_split, model_name)

    def _get_model(self, name):
        name = name.lower()
        model_map = {
            "srcnn": SRCNN(),
            "fsrcnn": FSRCNN(),
            "lapsr": LapSRN(in_channels=1, upscale_factor=1),
            "carnm": CARNM(num_channels=1, scale_factor=1),
            "falsrb": FALSRB(in_channels=1, out_channels=1, scale_factor=1),
            "srresnet": SRResNet(in_channels=1, out_channels=1, upscale_factor=1),
            "carn": CARN(in_channels=1, out_channels=1, upscale_factor=1),
            "falsra": FALSR_A(),
            "oisrrk2": OISRRK2(),
            "mdsr": MDSR(in_channels=1, upscale_factor=1, num_blocks=16),
            "san": SAN(in_channels=1, upscale_factor=1, num_blocks=16, num_heads=8),
            "rcan": RCAN(in_channels=1, num_blocks=1, upscale_factor=16),
            "unet": UNet(in_channels=1, out_channels=1),
            "dlgsanet": DLGSANet(in_channels=1, upscale_factor=1),
            "dpmn": DPMN(in_channels=1, upscale_factor=1),
            "safmn": SAFMN(in_channels=1, upscale_factor=1),
            "dpt": Net(angRes=5, factor=1),
            "distgssr": distgssr(angRes=5, factor=1),
            "swin": SwinIR(upscale=1, img_size=(self.patch_size, self.patch_size),
                           window_size=5, img_range=1., depths=[6, 6, 6, 6],
                           embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                           upsampler='pixelshuffledirect')
        }
        return model_map.get(name, None) or ValueError(f"Unknown model name: {name}")

    def _train(self, val_split, test_split, model_name):
        if self.use_wandb:
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

        x = self.input_da.values.astype(np.float32)  # (samples, y, x)
        y = self.target_da.values.astype(np.float32)

        # Apply patchify sample-by-sample
        x_patches = np.concatenate([patchify(img, self.patch_size) for img in x], axis=0)
        y_patches = np.concatenate([patchify(img, self.patch_size) for img in y], axis=0)

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

        best_val_loss = float("inf")
        best_model_state = None
        wait = 0

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

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {epoch} with best val_loss: {best_val_loss:.4f}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

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
                reconstructed = unpatchify(preds, x_input.shape, self.patch_size)
            else:
                x_tensor = torch.from_numpy(x_input[None, None, :, :]).to(self.device)
                pred = self.model(x_tensor).cpu().numpy()[0, 0, :, :] * self.y_max
                pred[pred < 0] = 0.0
                reconstructed = pred

        return xr.DataArray(reconstructed, coords=input_da.coords, dims=input_da.dims)
