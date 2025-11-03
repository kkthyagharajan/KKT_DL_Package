from setuptools import setup, find_packages

setup(
    name="KKT_DL_Package",
    version="1.0.0",
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
    install_requires=[
        "tensorflow==2.18.0",
        "opencv-python==4.11.0.0",
        "opencv-contrib-python==4.11.0.0",
        "pillow==11.1.0",
        "pyqt5==5.15.10",
        "cvlib==0.2.7",
        "numpy",
        "pandas",
        "seaborn",
        "matplotlib",
        "albumentations==1.3.1"
    ],
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

