#!/usr/bin/env python
import io
import os
import re
from datetime import datetime

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'prediff', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')

requirements = [
    'torch==2.0.1',
    'torchvision==0.15.2',
    'lightning>=2.0',
    'torchmetrics==1.2.0',
    'pandas',
    'h5py',
    'yacs',
    'omegaconf',
    'einops',
    'fvcore',
    'scipy',
    'scikit-learn',
    'ninja',
    'pillow',
    'opencv-python',
    'imageio',
    'scikit-image',
    'matplotlib',
    'tensorboard',
    'diffusers==0.13.0',
    'taming-transformers==0.0.1',
]

setup(
    # Metadata
    name='prediff',
    version=VERSION,
    python_requires='>=3.9',
    description='PreDiff: Precipitation Nowcasting with Latent Diffusion Models. '
                'Implementation in PyTorch-Lightning.',
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    # Package info
    packages=find_packages(
        where="src",
        exclude=(
            'tests',
            'scripts',
            'docs',)
    ),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
