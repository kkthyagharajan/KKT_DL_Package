from setuptools import setup, find_packages
from pathlib import Path
# Read dependencies from requirements.txt
this_dir = Path(__file__).parent
with open(this_dir / "requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="KKT_DL_Package",
    version="1.0.1",
    author="Thyagharajan K K",
    author_email="kkthyagharajan@yahoo.com",
    description=(
        "A Transfer and Deep Learning utility package for Keras and TensorFlow — including "
        "transfer learning, callbacks, dataset balancing, and visualization tools."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kkthyagharajan/KKT_DL_Package",
    packages=find_packages(),
    install_requires=requirements,  # automatically loads dependencies
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=(
        "deep learning, keras, tensorflow, transfer learning, "
        "callbacks, visualization, dataset tools, image processing"
    ),
    python_requires=">=3.11",
)
