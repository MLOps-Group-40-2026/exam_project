import pytest
import torch

from coffee_leaf_classifier.model import Model


def test_model_initialization():
    model = Model()
    assert isinstance(model, Model)


def test_model_forward_pass():
    model = Model()
    input_tensor = torch.randn(1, 3, 2048, 1024)  # Example input tensor
    output = model(input_tensor)
    assert output.shape[0] == 1  # Check batch size
    assert output.shape[1] == model.num_classes  # Check number of classes


def test_model_on_gpu():
    if torch.cuda.is_available():
        model = Model().cuda()
        input_tensor = torch.randn(1, 3, 2048, 1024).cuda()
        output = model(input_tensor)
        assert output.is_cuda
    else:
        pytest.skip("CUDA is not available")


def test_error_on_invalid_input():
    model = Model()
    invalid_input = torch.randn(1, 1, 1024, 2048)  # Invalid input shape
    with pytest.raises(RuntimeError):
        model(invalid_input)
