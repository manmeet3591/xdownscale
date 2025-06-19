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
  - name: Indian Institute of Technology, Roorkee, India
    index: 3
date: "2025-06-06"
bibliography: paper.bib
---

# Summary

`xdownscale` is an open-source Python package for spatiotemporal resolution enhancement (super-resolution) of gridded Earth observation datasets using deep learning. It provides a consistent and modular interface for applying and experimenting with a variety of deep learning models (e.g., UNet, SRCNN, FSRCNN) for downscaling geospatial data.

Built on top of `xarray`, `xdownscale` is designed for Earth science and remote sensing applications, enabling easy training, prediction, and GPU-accelerated workflows. The package simplifies patch-based inference and incorporates experiment tracking via Weights & Biases. It supports a variety of datasets such as DMSP-OLS and VIIRS nighttime lights, and land surface temperature (LST), helping users recover finer-resolution data from coarse measurements.

# Statement of Need

Remote sensing products are often spatially coarse due to sensor limitations or legacy archive resolutions. Deep learning-based super-resolution offers a promising avenue for increasing the utility of such datasets, but existing tools are often not tailored for geospatial data formats or workflows.

`xdownscale` fills this gap by:

- Supporting `xarray` data structures used in Earth science
- Providing access to state-of-the-art super-resolution architectures
- Enabling scalable, patch-based training and inference
- Supporting GPU acceleration and experiment tracking
- Reducing engineering overhead for geospatial deep learning experiments

It is intended for researchers and practitioners in climate science, remote sensing, and related disciplines.

# Installation

```bash
pip install git+https://github.com/manmeet3591/xdownscale.git
