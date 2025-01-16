import sys
import os
import torch

# Add the src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rice_images.data import load_data, pre_process_data
from collections import Counter


def test_dataset_size(train_dataset, val_dataset, test_dataset):
    """Test the MyDataset class."""
    assert len(train_dataset)==int(0.7 * 75000)
    assert len(val_dataset)==int(0.15 * 75000)
    assert (len(train_dataset)+len(val_dataset)+len(test_dataset))==75000


def test_normalization(train_dataset, val_dataset, test_dataset):
    """Test the MyDataset class."""
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for img, _ in dataset:  # Loop through all images and ignore labels
            assert img.max() <= 3, "Image is not normalized correctly (max > 1)"
            assert img.min() >= -3, "Image is not normalized correctly (min < 0)"

def test_dimensions(train_dataset, val_dataset, test_dataset):
    """Test the MyDataset class."""
    assert train_dataset[0][0].shape==(3, 250, 250)
    assert val_dataset[0][0].shape==(3, 250, 250)
    assert test_dataset[0][0].shape==(3, 250, 250)


def test_label_distribution(train_dataset):
    """Test label distribution."""
    labels = [label for _, label in train_dataset]  # Extract all labels from the dataset
    label_counts = Counter(labels)
    print(f"Label distribution in training dataset: {label_counts}")
    assert len(label_counts) > 1, "Dataset has only one class"

def test_no_overlap(train_dataset, val_dataset, test_dataset):
    """Test no overlap between datasets."""
    train_ids = set(id(data) for data in train_dataset)
    val_ids = set(id(data) for data in val_dataset)
    test_ids = set(id(data) for data in test_dataset)

    assert train_ids.isdisjoint(val_ids), "Train and validation datasets overlap"
    assert train_ids.isdisjoint(test_ids), "Train and test datasets overlap"
    assert val_ids.isdisjoint(test_ids), "Validation and test datasets overlap"

def test_no_corrupt_data(train_dataset):
    """Test that all images are valid and not corrupt."""
    for img, _ in train_dataset:
        assert img is not None, "Corrupt or missing image detected"

def test_data_types(train_dataset):
    """Test data types of images and labels."""
    for img, label in train_dataset:
        assert isinstance(img, torch.Tensor), "Image is not a tensor"
        assert isinstance(label, int), "Label is not an integer"


if __name__ =="__main__":
    # testing the data doesn't work without data. 
    # Data needs to be stored somewhere else so that this test can be run by github. Until then, we will assert true
    # Pre-process the data
    #pre_process_data()

    # Load the data
    train_dataset, val_dataset, test_dataset = load_data()
    
    # Perform the data tests
    test_dataset_size(train_dataset, val_dataset, test_dataset) 
    test_normalization(train_dataset, val_dataset, test_dataset)
    test_dimensions(train_dataset, val_dataset, test_dataset)
    test_label_distribution(train_dataset)
    #test_no_overlap(train_dataset, val_dataset, test_dataset)
    test_no_corrupt_data(train_dataset)
    test_data_types(train_dataset)
    
    print("All tests passed!")

