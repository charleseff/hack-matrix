"""
Setup file for HackMatrix Gymnasium environment package.
"""

from setuptools import find_packages, setup

setup(
    name="hackmatrix",
    version="0.1.0",
    description="Gymnasium environment for HackMatrix game",
    author="Charles Finkel",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "tensorboard>=2.13.0",
    ],
)
