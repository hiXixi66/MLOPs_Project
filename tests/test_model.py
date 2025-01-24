from rice_images.model import load_resnet18_timm
import os
import sys
import pytest
import torch

# Add the src folder to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)


@pytest.fixture(scope="module")
def model():
    """
    Pytest fixture to load the ResNet18 model.
    """
    return load_resnet18_timm()


def test_model_loading(model):
    """
    Test the model loading process.
    """
    assert isinstance(model, torch.nn.Module), "Model is not a PyTorch module."


def test_resnet_parameter_count(model):
    """
    Test the number of trainable parameters in the model.
    """
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_param_count = 11_179_077
    assert param_count == expected_param_count, (
        f"Parameter count mismatch. "
        f"Expected: {expected_param_count}, Got: {param_count}"
    )


def test_resnet_forward_pass(model):
    """
    Test the forward pass of the ResNet18 model.
    """
    model.eval()  # Set model to evaluation mode
    sample_input = torch.rand(
        1, 3, 224, 224
    )  # Example input: Batch=1, Channels=3, H=224, W=224
    output = model(sample_input)

    assert output.shape == (
        1,
        5,
    ), f"Output shape mismatch. Expected: (1, 5), Got: {output.shape}."
    assert not torch.any(torch.isnan(output)), "Output contains NaN values."


def test_resnet_layer_configuration(model):
    """
    Test the configuration of the ResNet layers.
    """
    # Test first convolutional layer
    assert model.conv1.kernel_size == (7, 7), "conv1 kernel size is incorrect."
    assert model.conv1.stride == (2, 2), "conv1 stride is incorrect."

    # Test the fully connected (fc) layer
    assert (
        model.fc.in_features == 512
    ), "Fully connected layer input size mismatch."
    assert (
        model.fc.out_features == 5
    ), "Fully connected layer output size mismatch."


def test_resnet_batchnorm(model):
    """
    Test the configuration of BatchNorm layers in the model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.affine is True, "BatchNorm layer is not affine."
            assert (
                module.track_running_stats is True
            ), "BatchNorm layer is not tracking running stats."


def test_resnet_output_range(model):
    """
    Test the output range of the ResNet18 model.
    """
    model.eval()
    sample_input = torch.rand(1, 3, 224, 224)
    output = model(sample_input)

    assert torch.all(output < 100), "Output values are too large."
    assert torch.all(output > -100), "Output values are too small."


def test_resnet_consistency(model):
    """
    Test the consistency of the model's output for the same input.
    """
    model.eval()
    sample_input = torch.rand(1, 3, 224, 224)
    output1 = model(sample_input)
    output2 = model(sample_input)

    assert torch.allclose(
        output1, output2, atol=1e-5
    ), "Model output is inconsistent across runs."


def test_resnet_gradient_flow(model):
    """
    Test the gradient flow in the ResNet18 model.
    """
    model.train()
    sample_input = torch.rand(1, 3, 224, 224, requires_grad=True)
    output = model(sample_input).sum()  # Sum to create a scalar output
    output.backward()

    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for a parameter."
