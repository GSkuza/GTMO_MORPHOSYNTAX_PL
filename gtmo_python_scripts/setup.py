#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Analysis System - Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh 
                   if line.strip() and not line.startswith("#")]

setup(
    name="gtmo-analysis",
    version="1.2.0",
    author="Grzegorz Skuza",
    author_email="",
    description="GTMØ Polish Morphosyntax Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gtmo-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: Polish",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gtmo-analyze=gtmo_morphosyntax:main",
            "gtmo-load=gtmo_file_loader:main",
            "gtmo-save=gtmo_json_saver:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)