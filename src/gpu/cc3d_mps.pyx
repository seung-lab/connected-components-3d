# cython: language_level=3

import torch
cimport torch

from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)

cdef extern from "cc3d_mps_impl.h":
    void connected_components_4_mps(
        uint64_t labelsAddr,
        uint64_t sx, uint64_t sy,
        uint64_t outputAddr,
        uint64_t osx, uint64_t osy
    ) except +

def connected_components_4(labels):
    if not labels.device.is_mps():
        raise RuntimeError("Unable to run non-mps tensor on an MPS device.")

    output = torch.zeros(
        labels.shape, dtype=torch.uint32, device='mps'
    )

    connected_components_4_mps(
        labels.data_ptr(), labels.shape[0], labels.shape[1],
        output.data_ptr(), output.shape[0], output.shape[1]
    )

    return output


