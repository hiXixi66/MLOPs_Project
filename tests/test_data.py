from collections import Counter

import pytest
import torch


from rice_images.data import load_data


@pytest.fixture(scope="module")
def datasets():
    """
    Fixture to load all datasets (train, validation, test).
    """
    train_dataset, val_dataset, test_dataset = load_data()
    return train_dataset, val_dataset, test_dataset


@pytest.fixture(scope="module")
def train_dataset(datasets):
    """
    Fixture for the training dataset.
    """
    return datasets[0]


@pytest.fixture(scope="module")
def val_dataset(datasets):
    """
    Fixture for the validation dataset.
    """
    return datasets[1]


@pytest.fixture(scope="module")
def test_dataset(datasets):
    """
    Fixture for the test dataset.
    """
    return datasets[2]


def test_dataset_size(train_dataset, val_dataset, test_dataset):
    """
    Test the size of each dataset and ensure the splits add up to the total.
    """
    assert len(train_dataset) == int(
        0.7 * 75_000
    ), "Training dataset size is incorrect."
    assert len(val_dataset) == int(
        0.15 * 75_000
    ), "Validation dataset size is incorrect."
    assert (
        len(train_dataset) + len(val_dataset) + len(test_dataset) == 75_000
    ), "Total dataset size is incorrect."


def test_normalization(train_dataset, val_dataset, test_dataset):
    """
    Test that images in all datasets are properly normalized.
    """
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for img, _ in dataset:
            assert (
                img.max() <= 3
            ), "Image is not normalized correctly (max > 3)."
            assert (
                img.min() >= -3
            ), "Image is not normalized correctly (min < -3)."


def test_dimensions(train_dataset, val_dataset, test_dataset):
    """
    Test the dimensions of images in all datasets.
    """
    expected_shape = (3, 250, 250)
    assert (
        train_dataset[0][0].shape == expected_shape
    ), "Training dataset image shape is incorrect."
    assert (
        val_dataset[0][0].shape == expected_shape
    ), "Validation dataset image shape is incorrect."
    assert (
        test_dataset[0][0].shape == expected_shape
    ), "Test dataset image shape is incorrect."


def test_label_distribution(train_dataset):
    """
    Test the distribution of labels in the training dataset.
    """
    labels = [label for _, label in train_dataset]
    label_counts = Counter(labels)
    print(f"Label distribution in training dataset: {label_counts}")
    assert len(label_counts) > 1, "Training dataset contains only one class."


def test_no_overlap(train_dataset, val_dataset, test_dataset):
    """
    Test that there is no overlap between train, validation, and test datasets.
    """
    train_ids = set(id(data) for data in train_dataset)
    val_ids = set(id(data) for data in val_dataset)
    test_ids = set(id(data) for data in test_dataset)

    assert train_ids.isdisjoint(
        val_ids
    ), "Train and validation datasets overlap."
    assert train_ids.isdisjoint(test_ids), "Train and test datasets overlap."
    assert val_ids.isdisjoint(
        test_ids
    ), "Validation and test datasets overlap."


def test_no_corrupt_data(train_dataset):
    """
    Test that all images in the training dataset are valid and not corrupt.
    """
    for img, _ in train_dataset:
        assert (
            img is not None
        ), "Corrupt or missing image detected in training dataset."


def test_data_types(train_dataset):
    """
    Test the data types of images and labels in the training dataset.
    """
    for img, label in train_dataset:
        assert isinstance(img, torch.Tensor), "Image is not a tensor."
        assert isinstance(label, int), "Label is not an integer."
