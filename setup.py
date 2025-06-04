from setuptools import setup, find_packages

setup(
    name='xdownscale',
    version='0.1.0',
    description='A PyTorch-based tool to downscale satellite images using SRCNN',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'torch',
        'xarray',
        'numpy',
    ],
)

