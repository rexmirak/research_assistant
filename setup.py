"""Setup configuration for research assistant."""

from setuptools import find_packages, setup

setup(
    name="research-assistant",
    version="0.1.1",
    description="Intelligent research paper analysis pipeline with local LLMs",
    author="Research Assistant Team",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["cli", "config", "check_install"],
    install_requires=[
        "click>=8.1.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "PyMuPDF>=1.23.0",
        "pdfminer.six>=20221105",
        "ocrmypdf>=15.0.0",
        "requests>=2.31.0",
        "habanero>=1.2.3",
        "datasketch>=1.6.0",
        "xxhash>=3.4.0",
        "ollama>=0.1.0",
        "google-generativeai>=0.3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "sqlalchemy>=2.0.0",
        "filelock>=3.12.0",
        "tqdm>=4.66.0",
        "coloredlogs>=15.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "commitizen>=3.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "research-assistant=cli:cli",
        ],
    },
)
