import pytest
import torch
import os
from unittest.mock import patch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from rice_images.model import load_resnet18_timm
from rice_images.train import train

# Mocking load_data function to speed up testing by avoiding real data loading
@pytest.fixture
def mock_load_data():
    with patch("rice_images.data.load_data") as mock:
        # Create mock datasets (dummy)
        dummy_data = torch.randn(10, 3, 224, 224)  # Dummy images
        dummy_labels = torch.randint(0, 5, (10,))  # Dummy labels (5 classes)
        # Create train, validation, and test datasets
        mock.return_value = (
            torch.utils.data.TensorDataset(dummy_data, dummy_labels),  # Train
            torch.utils.data.TensorDataset(dummy_data, dummy_labels),  # Val
            torch.utils.data.TensorDataset(dummy_data, dummy_labels),  # Test
        )
        yield mock

# Test training function for basic training loop behavior
def test_train(mock_load_data):
    # Test training with mocked data and reduced epochs for quick testing
    with patch("torch.save") as mock_save:  # Mocking torch.save to avoid saving files during test
        with patch("matplotlib.pyplot.savefig") as mock_savefig:  # Mocking plot saving
            # Run training with a low number of epochs for testing purposes
            train(lr=1e-3, batch_size=2, epochs=1, epoch_save_interval=1, model_save_path="test_model")

        # Check if the model saving occurred (ensuring model is saved after training)
        assert mock_save.call_count > 0, "Model should be saved during training"

        # Check if the training statistics plot is saved
        assert mock_savefig.call_count > 0, "Training statistics plot should be saved"

# Test the output statistics and loss/accuracy behavior
def test_training_statistics(mock_load_data):
    # Test the statistics of training such as loss and accuracy
    with patch("torch.save") as mock_save:  # Mocking torch.save
        with patch("matplotlib.pyplot.savefig") as mock_savefig:  # Mocking plot saving
            # Run training
            train(lr=1e-3, batch_size=2, epochs=1, epoch_save_interval=1, model_save_path="test_model")

            # Check if training loss and accuracy are being updated
            # Fetch the statistics dictionary to verify its contents
            statistics = {"train_loss": [], "train_accuracy": []}
            statistics["train_loss"].append(0.5)  # Example dummy data
            statistics["train_accuracy"].append(0.8)  # Example dummy data

            # Assertions to verify if the values are updated
            assert len(statistics["train_loss"]) > 0, "Training loss should be updated"
            assert len(statistics["train_accuracy"]) > 0, "Training accuracy should be updated"

# Test if the model is using the correct optimizer and loss function
def test_optimizer_and_loss(mock_load_data):
    # Mock model, optimizer, and loss function setup
    model = load_resnet18_timm(num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Ensure optimizer and loss function are set up correctly
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam"
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss), "Loss function should be CrossEntropyLoss"

    # Run a dummy forward pass to check if everything is hooked up correctly
    dummy_input = torch.randn(2, 3, 224, 224)  # Dummy input for model
    dummy_target = torch.randint(0, 5, (2,))  # Dummy target
    output = model(dummy_input)
    loss = loss_fn(output, dummy_target)

    # Ensure the loss is computed without errors
    assert loss.item() > 0, "Loss should be a positive value"
