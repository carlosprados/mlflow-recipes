# MLflowX Recipes - Extensible Fork

## Overview

Welcome to **MLflowX Recipes - Extensible Fork**, an extended and enhanced continuation of the original MLflow Recipes module. With MLflow's recent decision to discontinue support for Recipes, this project aims to pick up where the MLflow team left off. We not only maintain the existing functionality for classification and regression workflows but also reimagine Recipes as a more flexible and extensible framework for data scientists and developers. The primary goal is to empower you to create new recipe templates that can be easily incorporated, ensuring that MLflow remains a versatile and valuable tool for various machine learning workflows.

## Why This Fork?

The original MLflow Recipes module offered a simplified interface to streamline the model development process for common machine learning tasks, such as classification and regression. However, the lack of support for additional workflows and MLflow's decision to discontinue this feature left a gap for those who relied on Recipes for rapid prototyping and experimentation.

This project takes the core strengths of MLflow Recipes and extends its functionality:

- **Extensibility by Design**: Recipes are now modular and extensible, allowing developers and data scientists to easily add new workflows, such as time series forecasting, clustering, recommendation systems, and more.
- **Community-Driven Development**: By fostering a community-driven approach, this fork will continue to grow, adding diverse recipes contributed by developers worldwide.
- **Ease of Integration**: Maintaining compatibility with the existing MLflow ecosystem, this fork is designed to be easily integrated into your current projects with minimal changes to your workflows.

## Key Features

- **Support for Custom Recipes**: Unlike the original MLflow Recipes module, which only supported classification and regression, this fork allows users to easily develop and integrate custom recipes. Whether you need a workflow for anomaly detection, natural language processing, or another specialized task, this fork is built with modularity in mind.

- **Flexible Recipe Design**: Each recipe is designed to be self-contained and follows a consistent structure, making it easy for developers to understand and extend. This uniformity enables rapid onboarding and consistency across multiple machine learning projects.

- **Backward Compatibility**: Existing classification and regression recipes are maintained in their original form, ensuring that users migrating from MLflow's previous version will experience a seamless transition.

- **Community-Contributed Recipes**: We encourage the community to contribute their recipes to grow the library of supported workflows. We envision a collection of templates that cater to a wide range of machine learning challenges, all in one place.

## Getting Started

### Installation

To get started, you can install this fork as a Python package:

```bash
pip install git+https://github.com/carlosprados/mlflowx-recipes.git
```

Make sure you have MLflow installed:

```bash
pip install mlflow
```

### Usage

This fork follows the familiar `mlflow recipes` command-line interface, but with new commands and flags to accommodate extensibility. For example:

```bash
mlflow recipes run --recipe new_custom_recipe_name
```

You can also create and add your own recipes by following the template available in the `recipes/` directory:

```bash
recipes/
  |- custom_recipe_template/
      |- steps/
          |- ingest.py
          |- transform.py
          |- train.py
          |- evaluate.py
      |- recipe.yaml
```

### Creating a New Recipe

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/carlosprados/mlflowx-recipes.git
   ```

2. **Define Your Recipe Steps**:
   - Add your specific logic in the `steps/` directory.
   - Use the existing `recipe.yaml` as a starting point to define the workflow sequence.

3. **Run Your Recipe**:
   ```bash
   mlflow recipes run --recipe your_new_recipe_name
   ```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. We welcome contributions from the community to add new recipes, improve documentation, and enhance the core capabilities of the project.

To contribute:

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/new-recipe`).
3. Commit your changes (`git commit -m 'Add new recipe for time series forecasting'`).
4. Push to the branch (`git push origin feature/new-recipe`).
5. Open a pull request.

## License

This project is licensed under the terms of the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or suggestions, please reach out via the GitHub issues page or submit a pull request. We welcome all feedback and contributions that can help make this project more useful for the community.

---

We are excited to continue building upon MLflow's foundational work, with a vision to provide a more adaptable and expansive machine learning recipe framework. Let us know what recipes you'd like to see or contribute, and let's make this a truly versatile tool for all ML practitioners.
