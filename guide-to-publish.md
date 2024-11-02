# Guide to PyPI publishing

To create and publish your project as a Python package on PyPI, you'll need to follow these steps to prepare the repository and push it to PyPI effectively. Here is a step-by-step guide to help you through the process:

### 1. Create Necessary Files for Packaging
To package your project, you need to ensure that the following essential files are in place:

#### **`setup.py`**: The Setup Script
This file contains the metadata about your package, such as its name, version, author information, and more. Create a `setup.py` file with the following content:

```python
from setuptools import setup, find_packages

setup(
    name='mlflowx-recipes',
    version='0.1.0',
    author='Carlos Prados',
    author_email='your_email@example.com',
    description='An extensible fork of MLflow Recipes that allows easy integration of new machine learning workflows.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/carlosprados/mlflowx-recipes',
    packages=find_packages(),
    install_requires=[
        'mlflow>=1.25.0',  # Ensure compatibility with MLflow
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'mlflowx-recipes = mlflowx_recipes.cli:main',  # Example of adding a CLI entry point
        ],
    },
)
```
This script provides information about the package and defines dependencies that should be installed along with it.

#### **`setup.cfg`** (Optional): Configuration
You can also include `setup.cfg` for additional metadata and configuration, which can make `setup.py` cleaner:

```ini
[metadata]
description-file = README.md
```

#### **`MANIFEST.in`**: Include Additional Files
To include additional files like `README.md`, add a `MANIFEST.in` file:

```in
include README.md
include LICENSE
```

#### **Package Directory Structure**
Ensure that your Python code follows the correct structure. The main code should be in a dedicated directory, for example:

```
mlflowx-recipes/
  ├── mlflowx_recipes/           # Your package directory containing the code
  │   ├── __init__.py
  │   ├── cli.py                 # Optional: if you provide command-line functionality
  │   └── other_code_files.py
  ├── tests/                     # Include unit tests for your package
  │   ├── __init__.py
  │   └── test_something.py
  ├── README.md
  ├── setup.py
  ├── setup.cfg
  ├── MANIFEST.in
  └── LICENSE
```

### 2. Prepare Your Environment
You need to install tools that help you build and publish your Python package. First, make sure you're in your project's root directory, then run:

```bash
pip install setuptools wheel twine
```

- **`setuptools`** and **`wheel`** are used to create distribution packages.
- **`twine`** is used to upload the package to PyPI.

### 3. Create Distribution Archives
Now, create the source distribution and a wheel distribution:

```bash
python setup.py sdist bdist_wheel
```

- This will create a `dist/` directory with the `.tar.gz` source archive and `.whl` file.

### 4. Test Your Package Locally
Before publishing, it's good practice to install and test your package locally. Run:

```bash
pip install dist/mlflowx_recipes-0.1.0-py3-none-any.whl
```

Replace the file name as needed based on what was generated. Test whether the package functions as expected after the local installation.

### 5. Upload to Test PyPI (Optional, but Recommended)
To ensure everything works before uploading to the official PyPI, you can first upload to Test PyPI:

1. Create an account at [Test PyPI](https://test.pypi.org/account/register/).
2. Upload the package using `twine`:

   ```bash
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

3. You can then install it using:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ mlflowx-recipes
   ```

### 6. Publish to PyPI
Once you're ready to publish:

1. Create an account at [PyPI](https://pypi.org/account/register/).
2. Upload the package:

   ```bash
   twine upload dist/*
   ```

3. If successful, your package will now be available at `https://pypi.org/project/mlflowx-recipes/`.

### 7. Versioning and Updates
- Whenever you make changes to the project, be sure to update the version number in `setup.py`. For example, from `0.1.0` to `0.1.1`.
- Rebuild (`sdist bdist_wheel`) and re-upload (`twine upload`) the updated version to PyPI.

### 8. Best Practices
- **Testing**: Include unit tests (`tests/` folder) and consider setting up CI/CD pipelines (e.g., GitHub Actions) for automated testing.
- **Documentation**: Keep the `README.md` updated with installation instructions, examples, and usage. You could also create more detailed documentation using platforms like [Read the Docs](https://readthedocs.org/).
- **Community Guidelines**: Create a `CONTRIBUTING.md` file to guide others on how to contribute effectively, and optionally, a `CODE_OF_CONDUCT.md` to set expectations.

### Example CLI Usage
If you add the `entry_points` in `setup.py`, users who install your package will be able to run commands like:

```bash
mlflowx-recipes run --recipe new_custom_recipe_name
```

This guide should cover all the necessary steps for transforming your `mlflowx-recipes` project into a package ready for PyPI publication. Let me know if you need more detailed instructions on any particular step, Charlie!
