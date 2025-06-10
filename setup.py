from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='xdownscale',
    version='1.0.1',
    description='A PyTorch-based tool to downscale spatiotemporal data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Manmeet Singh, Naveen Sudharsan',
    url='https://github.com/manmeet3591/xdownscale',
    packages=find_packages(),
    install_requires=[
        'torch',
        'xarray',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
