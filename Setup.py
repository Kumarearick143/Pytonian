# setup.py
from setuptools import setup, find_packages

setup(
    name="Pytonian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "numpy",
        "gudhi",
        "matplotlib"
    ],
    extras_require={
        "quantum": ["qiskit", "pennylane"]
    }
)