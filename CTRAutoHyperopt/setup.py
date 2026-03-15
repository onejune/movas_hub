"""Setup script for CTR Auto HyperOpt."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ctr-auto-hyperopt",
    version="0.1.0",
    author="Walter Wan",
    author_email="walter.wan@example.com",
    description="Automated Model Selection & Hyperparameter Optimization for CTR Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ctr_auto_hyperopt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=2.0",
        "scikit-learn>=1.0",
        "torch>=2.0",
        "optuna>=3.0",
        "flaml>=2.0",
        "deepctr-torch>=0.2.9",
        "lightgbm>=4.0",
        "xgboost>=2.0",
    ],
    extras_require={
        "full": [
            "tensorflow>=2.10",
            "catboost>=1.2",
            "mlgb>=0.8.0",
        ],
    },
)
