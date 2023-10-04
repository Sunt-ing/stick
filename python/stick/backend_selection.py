"""Backend selection between numpy and self-impl NDArray."""
import os


BACKEND = os.environ.get("STICK_BACKEND", "nd")


if BACKEND == "nd":
    from .backend_ndarray import ndarray as array_api

    cpu = array_api.cpu
    cuda = array_api.cuda
    cpu_numpy = array_api.cpu_numpy
    all_devices = array_api.all_devices
    default_device = array_api.default_device
    Device = array_api.BackendDevice
    NDArray = array_api.NDArray

elif BACKEND == "np":
    import numpy as array_api
    from . import backend_numpy

    cpu = backend_numpy.cpu
    cuda = None
    all_devices = backend_numpy.all_devices
    default_device = backend_numpy.default_device
    Device = backend_numpy.Device

    NDArray = array_api.NDArray
else:
    raise RuntimeError("Unknown stick array backend %s" % BACKEND)
