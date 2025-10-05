"""
Setup configuration for context-aware-semantic-testcase-search project.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="context-aware-semantic-testcase-search",
    version="1.0.0",
    description="Semantic clustering and visualization of software test cases using ELMo, BERT, and T5 embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rvaghashiya/context-aware-semantic-testcase-search",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.20.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "semantic-cluster=src.cli:main",
        ],
    },
)