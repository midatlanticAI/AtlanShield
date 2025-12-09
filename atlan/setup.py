from setuptools import setup, find_packages

setup(
    name="atlan-core",
    version="0.1.0",
    description="The Resonant Cognitive Architecture (RCA) Core",
    author="Atlan Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "psutil"
    ],
    python_requires=">=3.8",
)
