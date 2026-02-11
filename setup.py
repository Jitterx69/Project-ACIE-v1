from setuptools import setup, find_packages

setup(
    name="acie",
    version="0.1.0",
    description="Astronomical Counterfactual Inference Engine",
    author="ACIE Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "numpy>=1.21.0,<2.0",  # Relaxed for compatibility
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "networkx>=2.8",  # Relaxed for compatibility
        "pyro-ppl>=1.8.0",  # For probabilistic programming
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.12.0",
        # Development dependencies
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "acie=acie.cli:main",
        ],
    },
    extras_require={
        "sdk": [
            "requests>=2.31.0",
        ],
        "cli": [
            "typer[all]>=0.9.0",
            "rich>=13.0.0",
        ],
        "all": [
            "requests>=2.31.0",
            "typer[all]>=0.9.0",
            "rich>=13.0.0",
        ],
    },
)
