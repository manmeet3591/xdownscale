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

`xdownscale`[@manmeet2025xdownscale] is an open-source Python package for spatiotemporal resolution enhancement (super-resolution) of gridded Earth observation datasets using deep learning. It provides a consistent and modular interface for applying and experimenting with a variety of deep learning models (e.g., UNet, SRCNN, FSRCNN) for downscaling geospatial data.

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

# Mathematics

Super-resolution models often minimize a loss function $L$ over the predicted high-resolution image $\hat{Y}$ and the true image $Y$, typically defined as:

$$
L(\hat{Y}, Y) = \frac{1}{n} \sum_{i=1}^n (\hat{Y}_i - Y_i)^2
$$

where $n$ is the number of pixels. More advanced models may also include perceptual or adversarial loss components.

# Citations

This package is developed and distributed via GitHub [@manmeet2025xdownscale]. For deep learning model architectures like UNet and SRCNN, we recommend citing their respective original papers.

For a quick reference, the following citation commands can be used:
- `@author:2024`  →  "Author et al. (2024)"
- `[@author:2024]` → "(Author et al., 2024)"
- `[@a:2024; @b:2023]` → "(A et al., 2024; B et al., 2023)"

# Figures

Figures can be included like this:

![Example architecture of the super-resolution model.\label{fig:model}](architecture.png)

And referenced from text using \autoref{fig:model}.

You can also scale the figure size:

![Dataset visualization.](dataset.png){ width=40% }

# Acknowledgements

We thank the contributors to the open-source software ecosystem, especially the developers of PyTorch, `xarray`, and Weights & Biases. We are grateful for institutional support from The University of Texas at Austin and the Indian Institute of Technology Roorkee. This research was partially supported by research computing facilities and grants from the authors' institutions.

# References
