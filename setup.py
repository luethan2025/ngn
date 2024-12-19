#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
  name="ngn",
    version='0.1',
    packages=find_packages(include=["common"]),
    package_dir={
      "common": "common"
  }
)
