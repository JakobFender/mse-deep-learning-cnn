import pytest
import torch

from src.model.cnn import CNN


@pytest.fixture
def default_model():
    return CNN()


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(4, 3, 128, 128)  # (batch, RGB, H, W)


def test_output_shape_default(default_model, input_tensor):
    output = default_model(input_tensor)
    assert output.shape == (4, 10)


@pytest.mark.parametrize("num_classes", [2, 10, 100])
def test_output_shape_num_classes(num_classes, input_tensor):
    model = CNN(num_classes=num_classes)
    output = model(input_tensor)
    assert output.shape == (4, num_classes)


@pytest.mark.parametrize("channels", [(32,), (32, 64), (32, 64, 128), (16, 32, 64, 128)])
def test_output_shape_variable_blocks(channels, input_tensor):
    model = CNN(channels=channels)
    output = model(input_tensor)
    assert output.shape == (4, 10)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_output_shape_batch_sizes(batch_size):
    model = CNN()
    x = torch.randn(batch_size, 3, 128, 128)
    output = model(x)
    assert output.shape == (batch_size, 10)


def test_output_is_raw_logits(default_model, input_tensor):
    """Output should be raw logits, not probabilities — values outside [0, 1] expected."""
    output = default_model(input_tensor)
    assert not torch.all((output >= 0) & (output <= 1)), \
        "Output looks like probabilities; expected raw logits."


def test_output_no_nan(default_model, input_tensor):
    output = default_model(input_tensor)
    assert not torch.isnan(output).any()


def test_assert_empty_channels():
    with pytest.raises(AssertionError):
        CNN(channels=())


def test_assert_too_many_pool_layers():
    """8 pool layers of size 2 on a 128px image collapses spatial dims to 0."""
    with pytest.raises(AssertionError):
        CNN(channels=(32,) * 8, pool_size=2, input_size=128)


def test_dropout_disabled_in_eval(default_model, input_tensor):
    """Repeated eval-mode passes must be deterministic."""
    default_model.eval()
    with torch.no_grad():
        out1 = default_model(input_tensor)
        out2 = default_model(input_tensor)
    assert torch.allclose(out1, out2)


def test_dropout_active_in_train():
    """With high dropout, repeated train-mode passes should differ."""
    model = CNN(dropout_p=0.9)
    model.train()
    x = torch.randn(16, 3, 128, 128)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)


def test_no_dropout_layer_when_p_zero():
    model = CNN(dropout_p=0.0)
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.classifier)
    assert not has_dropout
