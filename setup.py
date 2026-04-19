from setuptools import setup, find_packages

setup(
    name="diffuspec",
    version="0.1.0",
    description="DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.47.0",
        "accelerate>=0.34.0",
        "numpy>=1.24.0",
        "tqdm",
    ],
)
