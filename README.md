# rice_images

rice classifier with a focus on MLOps pipeline

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Project Description

### Goal

This is the project description for Group 18 in the Machine Learning Operations course at DTU. The primary objective of this project is to apply the concepts and techniques learned throughout the course to a real-world machine learning problem: classifying different types of rice from images. To approach this, we plan to leverage the PyTorch Image Models (TIMM) framework, which offers a collection of pre-trained models and tools that can significantly streamline the development and fine-tuning process for image classification tasks. Finally, we will present our findings through a concise presentation and submit the completed code for evaluation.

### Framework
For this project, we are leveraging the PyTorch Image Models (TIMM) library, a powerful open-source tool designed to simplify and enhance image classification tasks. TIMM provides a wide range of pre-trained models, including state-of-the-art architectures like ResNet, EfficientNet, and Vision Transformers, which we will use to experiment and optimize our classification pipeline. By incorporating TIMM, we can utilize pre-trained weights for transfer learning, significantly reducing training time and improving model performance. The library seamlessly integrates with PyTorch, allowing us to include its models and utilities in our training and evaluation pipeline. Initially, we will fine-tune pre-trained ResNet models to establish a strong baseline and then experiment with other architectures based on performance. TIMM also offers tools for customizing and fine-tuning models, which will enable us to adapt the models specifically for the rice image dataset.

### Data
The data we have chosen to work with consists of 75,000 images of rice grains, divided into five categories, with 15,000 images per category. The dataset, titled "Rice Image Dataset," can be accessed via the following link: https://www.muratkoklu.com/datasets/. The images will be normalized using the mean and standard deviations of the ImageNet dataset, as we will utilize pre-trained weights from the ResNet architecture.

### Models

We aim to perform an image classification task on the rice dataset using convolutional neural networks (CNNs). Our approach will include:

1.  ResNet: Leveraging pre-trained models from the TIMM framework for robust feature extraction and fine-tuning.
2.  Baseline CNN: Implementing a simple convolutional architecture as a baseline to compare performance against ResNet.

In addition to experimenting with these models, we plan to explore and tune various hyperparameters such as learning rate, optimizer, batch size, and model depth to optimize classification performance.
