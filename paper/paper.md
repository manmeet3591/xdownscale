---
title: 'xdownscale: A Deep Learning Toolkit for Spatiotemporal Resolution Enhancement of Gridded Data'
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
    affiliation: 3
affiliations:
  - name: The University of Texas at Austin
    index: 1
  - name: Leibniz Centre for Agricultural Landscape Research (ZALF), Müncheberg, Germany
    index: 2
date: 2025-06-06
bibliography: paper.bib
---

# Summary

`xdownscale` is an open-source Python package for spatial downscaling of gridded Earth observation datasets using deep learning. It supports a wide range of super-resolution models, including UNet, SRCNN, FSRCNN, and others, with a consistent interface for training and prediction using `xarray`-based inputs. Designed for researchers and practitioners in remote sensing and environmental science, `xdownscale` simplifies model selection, training configuration, patch-based inference, GPU usage, and experiment tracking (via Weights & Biases).

Its modular design allows rapid experimentation and deployment of deep learning-based downscaling on datasets such as DMSP and VIIRS nighttime lights, land surface temperature, and more.

# Statement of Need

Remote sensing datasets often suffer from spatial limitations due to sensor constraints or archival resolutions. Super-resolution methods using deep learning have shown promise but lack standardized tools in geospatial formats. `xdownscale` addresses this gap by:

- Supporting domain-specific data structures (`xarray`)
- Providing multiple state-of-the-art models
- Enabling scalable patch-based training
- Simplifying deployment on GPUs
- Offering logging and visualization via Weights & Biases

It serves Earth scientists, climate modelers, urban analysts, and researchers aiming to reconstruct high-resolution signals from coarser sources.

# Installation

```bash
pip install git+https://github.com/manmeet3591/xdownscale.git
