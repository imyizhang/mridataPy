#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()

setuptools.setup(
    name='mridataPy',
    version='0.0.6',
    description='A lightweight toolbox for downloading and processing mridata from mridata.org',
    author='Yi Zhang',
    author_email='yizhang.dev@gmail.com',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/yzhang-dev/mridataPy',
    download_url='https://github.com/yzhang-dev/mridataPy',
    packages=setuptools.find_packages(),
    keywords=[
        'mri', 'mri-reconstruction', 'mridata'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    license='MIT',
    python_requires='>=3.8',
    install_requires=[
        'ismrmrd>=1.7.1',
        'numpy>=1.18.5',
        'numba>=0.53.0',
        'requests>=2.25.1',
        'scipy>=1.5.0',
        'tqdm>=4.48.0',
    ],
)
