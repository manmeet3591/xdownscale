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
  - name: The University of Texas at Austin
    index: 1
  - name: Leibniz Centre for Agricultural Landscape Research (ZALF), Müncheberg, Germany
    index: 2
date: 2025-06-06
bibliography: paper.bib
---

# Summary

`xdownscale` is an open-source Python toolkit for super-resolution of gridded geospatial datasets using deep learning. It enables the enhancement of spatiotemporal resolution in Earth observation and climate data by providing a unified interface to train and apply deep learning models such as UNet, SRCNN, FSRCNN, and others.

Designed with Earth science applications in mind, `xdownscale` builds on the `xarray` ecosystem and supports patch-based training, GPU acceleration, and experiment tracking with Weights & Biases. It is particularly well-suited for downscaling satellite-derived products like nighttime lights (e.g., DMSP-OLS, VIIRS) and land surface temperature (LST), among others.

The package offers an efficient workflow for data preparation, model training, and evaluation, making it easy for researchers to apply and benchmark super-resolution techniques on geospatial grids.

# Statement of Need

Many remote sensing and climate datasets are available at coarse spatial resolutions due to sensor limitations or archival constraints. Enhancing the resolution of these datasets is crucial for fine-scale environmental monitoring, urban studies, and climate adaptation research.

While deep learning-based super-resolution methods have proven effective, existing tools are rarely optimized for the structure and scale of geospatial data. `xdownscale` addresses this gap by offering:

- Native support for `xarray '- based workflows, commonly used in geoscience
- Integration of multiple state-of-the-art deep learning architectures
- Scalable, patch-based inference and training strategies
- Seamless GPU acceleration and optional logging with Weights & Biases
- A modular design that simplifies experimentation and benchmarking

By abstracting away low-level engineering complexity, `xdownscale` empowers researchers and practitioners in Earth and environmental sciences to easily apply and extend super-resolution techniques to geospatial applications.

# Installation

```bash
pip install git+https://github.com/manmeet3591/xdownscale.git
