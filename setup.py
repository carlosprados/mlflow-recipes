from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="training-plan-template",
    version="0.1.0",
    author="Carlos Prados",
    author_email="carlos.prados@amplia.es",
    description="An extensible fork of MLflow Recipes that allows easy integration of new machine learning workflows.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/carlosprados/opengate-ai-recipes",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "training-plan-template = opengate.recipes.cli:commands",  # Example of adding a CLI entry point
        ],
    },
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)
