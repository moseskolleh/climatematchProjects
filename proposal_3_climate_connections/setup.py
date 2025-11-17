"""
Setup script for Climate Teleconnection Discovery System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="climate_teleconnections",
    version="0.1.0",
    author="Moses Kolleh",
    author_email="moses@example.com",
    description="Theory-Guided Discovery of Climate System Connections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moseskolleh/climatematchProjects",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.1.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tc-discover=discovery.cli:discover",
            "tc-validate=validation.cli:validate",
            "tc-monitor=operational.cli:monitor",
        ],
    },
)
