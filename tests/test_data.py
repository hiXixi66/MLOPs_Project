import sys
import os
import torch
import pytest

# Add the src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rice_images.data import load_data, pre_process_data
from collections import Counter

# Fixture to load the datasets
@pytest.fixture(scope="module")
def datasets():
    train_dataset, val_dataset, test_dataset = load_data()
    return train_dataset, val_dataset, test_dataset

@pytest.fixture(scope="module")
def train_dataset(datasets):
    return datasets[0]

@pytest.fixture(scope="module")
def val_dataset(datasets):
    return datasets[1]

@pytest.fixture(scope="module")
def test_dataset(datasets):
    return datasets[2]

# Test the size of the dataset.
def test_dataset_size(train_dataset, val_dataset, test_dataset):
    assert len(train_dataset)==int(0.7 * 75000)
    assert len(val_dataset)==int(0.15 * 75000)
    assert (len(train_dataset)+len(val_dataset)+len(test_dataset))==75000

# Test the normalization of the images in the dataset.
def test_normalization(train_dataset, val_dataset, test_dataset):
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for img, _ in dataset:  # Loop through all images and ignore labels
            assert img.max() <= 3, "Image is not normalized correctly (max > 1)"
            assert img.min() >= -3, "Image is not normalized correctly (min < 0)"

# Test the dimensions of the images in the dataset.
def test_dimensions(train_dataset, val_dataset, test_dataset):
    assert train_dataset[0][0].shape==(3, 250, 250)
    assert val_dataset[0][0].shape==(3, 250, 250)
    assert test_dataset[0][0].shape==(3, 250, 250)

# Test the label distribution in the dataset.
def test_label_distribution(train_dataset):
    labels = [label for _, label in train_dataset]  # Extract all labels from the dataset
    label_counts = Counter(labels)
    print(f"Label distribution in training dataset: {label_counts}")
    assert len(label_counts) > 1, "Dataset has only one class"

# Test that there is no overlap between datasets.
def test_no_overlap(train_dataset, val_dataset, test_dataset):
    train_ids = set(id(data) for data in train_dataset)
    val_ids = set(id(data) for data in val_dataset)
    test_ids = set(id(data) for data in test_dataset)

    assert train_ids.isdisjoint(val_ids), "Train and validation datasets overlap"
    assert train_ids.isdisjoint(test_ids), "Train and test datasets overlap"
    assert val_ids.isdisjoint(test_ids), "Validation and test datasets overlap"

# Test that all images are valid and not corrupt.
def test_no_corrupt_data(train_dataset):
    for img, _ in train_dataset:
        assert img is not None, "Corrupt or missing image detected"

# Test data types of images and labels.
def test_data_types(train_dataset):
    for img, label in train_dataset:
        assert isinstance(img, torch.Tensor), "Image is not a tensor"
        assert isinstance(label, int), "Label is not an integer"
