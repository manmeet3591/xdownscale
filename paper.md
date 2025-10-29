---
title: "xdownscale: A Deep Learning Toolkit for Spatiotemporal Resolution Enhancement of Gridded Data"
tags:
  - Python
  - deep learning
  - super-resolution
  - satellite data
  - geospatial
  - remote sensing
authors:
  - name: Manmeet Singh
    orcid: 0000-0002-3374-7149
    affiliation: 1
    corresponding: true
  - name: Naveen Sudharsan
    orcid: 0000-0002-1328-110X
    affiliation: 1
  - name: Hassan Dashtian
    orcid: 0000-0001-6400-1190
    affiliation: 1
  - name: Harsh Kamath
    orcid: 0000-0002-5210-8369
    affiliation: 1
  - name: Amit Kumar Srivastava
    orcid: 0000-0001-8219-4854
    affiliation: 2
affiliations:
  - name: The University of Texas at Austin, USA
    index: 1
  - name: Leibniz Centre for Agricultural Landscape Research (ZALF), Müncheberg, Germany
    index: 2
date: "2025-06-19"
bibliography: paper.bib
---

# Summary

**xdownscale** is an open-source Python toolkit for super-resolution downscaling of gridded datasets using deep learning. It provides a unified interface to train and apply a variety of convolutional neural network models (e.g., SRCNN, FSRCNN, SRResNet, RCAN, U-Net, SwinIR) for enhancing the spatial and temporal resolution of Earth science data:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}. Built on **PyTorch** and the **xarray** data model, xdownscale efficiently maps from coarse to fine resolution in just a few lines of code while preserving metadata and coordinates:contentReference[oaicite:2]{index=2}. The toolkit supports patch-based training, multi-GPU acceleration, and optional experiment tracking via Weights & Biases, making it easy to train models on large geospatial datasets.

Originally designed for Earth observation and climate applications, xdownscale is well-suited for downscaling satellite products like nighttime light imagery and land surface temperature. For example, users can super-resolve historical low-resolution nighttime light data (e.g., DMSP-OLS at ~1 km) to produce finer-resolution imagery comparable to newer VIIRS data (~500 m), helping to reveal intra-urban details and trends. Likewise, xdownscale can enhance coarse climate model outputs or reanalysis data to local scales, aiding tasks such as urban heat island analysis or watershed hydrology studies. By providing built-in implementations of state-of-the-art super-resolution architectures, xdownscale lowers the barrier for researchers to apply advanced deep learning methods to these domain problems.

The toolkit emphasizes ease of use and extensibility. It integrates with the xarray ecosystem (widely used in geoscience data analysis:contentReference[oaicite:3]{index=3}) so that inputs and outputs are handled as `xarray.DataArray` objects. This allows seamless alignment of predictions with the original geospatial coordinates and time indices. In addition, xdownscale’s modular design encourages experimentation: users can select or customize neural network architectures, adjust training parameters, and compare results across models consistently. Overall, **xdownscale** accelerates the adoption of deep learning-based super-resolution in Earth sciences by abstracting away low-level details and providing an efficient end-to-end downscaling workflow.

# Statement of Need

Many remote sensing and climate datasets are available only at coarse spatiotemporal resolutions due to sensor limitations or storage constraints. Enhancing the resolution of these datasets is crucial for fine-scale environmental monitoring, urban studies, and climate impact assessments. Traditional statistical downscaling techniques can improve resolution but often require extensive hand-crafted modeling and may not fully capture complex spatial patterns:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}. In recent years, **deep learning** methods have demonstrated remarkable success in super-resolving images and gridded data:contentReference[oaicite:6]{index=6}. However, existing deep-learning super-resolution tools are rarely optimized for the structure and scale of geospatial data (e.g., handling latitude–longitude grids, time series, and large arrays). This gap has made it difficult for geoscientists to leverage advanced super-resolution models in their research.

**xdownscale** addresses this need by offering a domain-focused toolkit with the following key features:

- **Native xarray support:** It works directly with xarray data structures:contentReference[oaicite:7]{index=7}, which are ubiquitous in Earth science, allowing users to keep metadata and coordinate information attached through the downscaling process (e.g., retaining geo-referencing of satellite images):contentReference[oaicite:8]{index=8}.
- **Multiple architectures integrated:** The package includes implementations of numerous state-of-the-art CNN models for super-resolution, from early designs like SRCNN:contentReference[oaicite:9]{index=9} and FSRCNN:contentReference[oaicite:10]{index=10} to more advanced networks such as residual networks and transformer-based models (e.g., RCAN and SwinIR):contentReference[oaicite:11]{index=11}. This breadth enables users to easily compare and find the best approach for their data.
- **Scalable training and inference:** xdownscale uses patch-wise training and efficient data loaders to handle large gridded datasets that do not fit into memory. It supports training on GPUs and distributed computing, which is important given the size of typical climate or remote sensing data cubes.
- **Experiment tracking and reproducibility:** Built-in hooks for logging with Weights & Biases allow users to monitor training progress and record model parameters and metrics. This promotes reproducible research when developing new downscaling models.
- **Modularity and extensibility:** The code is organized so that researchers can plug in custom model architectures or loss functions without modifying the entire framework. This design facilitates extending the toolkit for novel experiments and ensures that it can keep pace with developments in deep learning.

By abstracting away low-level details (data tiling, model plumbing, etc.), xdownscale empowers researchers and practitioners in Earth and environmental sciences to focus on the scientific questions at hand. For instance, using xdownscale, one can train a model to downscale daily precipitation fields from a coarse global climate model (~100 km resolution) to a local 10 km grid, following approaches like DeepSD (which first applied super-resolution CNNs to climate data:contentReference[oaicite:12]{index=12}). The toolkit thus serves as a bridge between cutting-edge machine learning techniques and pressing geospatial problems, fulfilling a growing need for accessible, high-quality downscaling software in the scientific community.

# Implementation and Example

xdownscale is implemented in Python and available under the MIT open-source license. The core library relies on PyTorch (for model definitions and training loops) and xarray (for data management). Each model architecture is defined in a modular way, and the `Downscaler` class provides a simple interface for training and applying these models. After installation, users can immediately apply a pretrained model or train a new model with just a few lines of code:

```python
from xdownscale import Downscaler
# X and Y are xarray DataArray objects: X = low-res input, Y = high-res target
ds = Downscaler(X, Y, model_name="fsrcnn", epochs=50, learning_rate=1e-4)
ds.train()               # train the FSRCNN model on the provided data
Y_pred = ds.predict(X)   # run the trained model to get high-res prediction
