import pytest
import torch


def available_devices():
    """Return every device the running machine can actually use.

    CPU is always present. Accelerators are appended only when their backend
    reports as available, so the same test suite exercises CUDA, MPS, or XPU on
    machines that have them while still running everywhere else.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        devices.append("mps")
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        devices.append("xpu")
    return devices


@pytest.fixture(params=available_devices())
def device(request):
    return torch.device(request.param)
