from stage02.backend import TorchNPUBackend
from stage02.backend import TorchCPUBackend

def test_cpu_backend_tensor_device():
    backend = TorchCPUBackend()
    tensor = backend.tensor([1, 2, 3], dtype=torch.long)

    assert backend.is_available()
    assert tensor.device.type == "cpu"

def test_get_backend_cpu():
    backend = get_backend("cpu")
    assert backend.name == "cpu"
    assert backend.device.type == "cpu"

def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_backend("cuda")

