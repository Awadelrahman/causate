from setuptools import setup, find_packages

setup(
    name="causate",
    version="0.1.0",  # Increment the version for the new release
    description="Causate is a Python package designed to operationalize causal model discovery and inference. It enables seamless discovery, visualization, and deployment of causal relationships into actionable workflows.",
    long_description="Causate is a Python package designed to operationalize causal model discovery and inference. It enables seamless discovery, visualization, and deployment of causal relationships into actionable workflows.",  # Ensure you have a README.md file
    long_description_content_type="text/markdown",
    author="Awadelrahman M. A. Ahmed",
    author_email="awadrahman@gmail.com",
    url="https://github.com/Awadelrahman/causate",  # Update if needed
    packages=find_packages(),  # Automatically find and include all packages/modules
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
    "mlflow",  
    "pandas",
    "gcastle",  
    "torch",
    "numpy",  
    "networkx",  
    "matplotlib", 
],
)
