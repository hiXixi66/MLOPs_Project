import sys
import os
import torch

# Add the src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rice_images.model import load_resnet18_timm

# Test the model loading.
def test_model_loading(model):
    assert isinstance(model, torch.nn.Module), "Model is not a PyTorch module"    

# Test the number of parameters in the model.
def test_resnet_parameter_count(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_param_count = 11179077
    assert param_count == expected_param_count, f"Parameter count mismatch. Expected: {expected_param_count}, Got: {param_count}"

# Test the forward pass of the model.
def test_resnet_forward_pass(model):
    model.eval()  # Set model to evaluation mode
    sample_input = torch.rand(1, 3, 224, 224)  # Example input shape: Batch=1, Channels=3, H=224, W=224
    output = model(sample_input)

    assert output.shape == (1, 5), f"Output shape mismatch. Expected: (1, 5), Got: {output.shape}"
    assert not torch.any(torch.isnan(output)), "Output contains NaN values."

# Test the configuration of the ResNet layers.
def test_resnet_layer_configuration(model):
 
    # Test first convolutional layer
    assert model.conv1.kernel_size == (7, 7), "conv1 kernel size is incorrect."
    assert model.conv1.stride == (2, 2), "conv1 stride is incorrect."

    # Test the fully connected (fc) layer
    assert model.fc.in_features == 512, "Fully connected layer input size mismatch."
    assert model.fc.out_features == 5, "Fully connected layer output size mismatch."

# Test the configuration of the BatchNorm layers.
def test_resnet_batchnorm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.affine is True, "BatchNorm layer is not affine."
            assert module.track_running_stats is True, "BatchNorm layer is not tracking running stats."

# Test the output range of the model.
def test_resnet_output_range(model):
    model.eval()
    sample_input = torch.rand(1, 3, 224, 224)
    output = model(sample_input)

    assert torch.all(output < 100), "Output values are too large."
    assert torch.all(output > -100), "Output values are too small."

# Test the consistency of the model output.
def test_resnet_consistency(model):
    model.eval()
    sample_input = torch.rand(1, 3, 224, 224)
    output1 = model(sample_input)
    output2 = model(sample_input)

    assert torch.allclose(output1, output2, atol=1e-5), "Model output is inconsistent across runs."

# Test the gradient flow in the model.
def test_resnet_gradient_flow(model):
    model.train()
    sample_input = torch.rand(1, 3, 224, 224, requires_grad=True)
    output = model(sample_input).sum()  # Sum to create a scalar output
    output.backward()

    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for a parameter."

if __name__ == "__main__":
    model = load_resnet18_timm()

    test_model_loading(model)
    test_resnet_parameter_count(model)
    test_resnet_forward_pass(model)
    test_resnet_layer_configuration(model)
    test_resnet_batchnorm(model)
    test_resnet_output_range(model)
    test_resnet_consistency(model)
    test_resnet_gradient_flow(model)

    print("All tests passed!")