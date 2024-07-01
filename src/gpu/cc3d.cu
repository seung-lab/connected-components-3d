/*
 * This code was inspired by https://github.com/zsef123/Connected_components_PyTorch
 * and had some assistance from ChatGPT (I've never written CUDA kernels before).
 *
 * The license for the above code is BSD-3.
 * 
 * William Silversmith
 * June 30, 2024
 */

#ifndef __CC3D_CUDA__
#define __CC3D_CUDA__

#include <cuda.h>
#include <torch/script.h>

namespace cc3dGPU {

template <typename LABEL>
__global__ void init_labels(
    LABEL* labels, uint32_t* out_labels, 
    const uint32_t sx, const uint32_t sy
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t index = x + y * sx;
    
    out_labels[index] = 0;
    if (x < sx && y < sy && labels[index] > 0) {
        out_labels[index] = index;
    }
}

__global__ void union_find(int* labels, int sx, int sy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * sx + x;

    if (x >= sx || y >= sy) return;

    int label = labels[index];
    int min_label = label;

    // Check neighbors and perform union
    if (x > 0) min_label = min(min_label, labels[index - 1]);
    if (y > 0) min_label = min(min_label, labels[index - sx]);
    if (x < sx - 1) min_label = min(min_label, labels[index + 1]);
    if (y < sy - 1) min_label = min(min_label, labels[index + sx]);

    if (min_label < label) {
        labels[index] = min_label;
    }
}

__global__ void path_compression(int* labels, int sx, int sy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * sx + x;

    if (x >= sx || y >= sy) return;

    int label = labels[index];
    while (label != labels[label]) {
        label = labels[label];
    }
    labels[index] = label;
}

template <typename LABEL>
void connected_components_4(
    LABEL* labels, uint32_t* out_labels, int sx, int sy
) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (sx + blockSize.x - 1) / blockSize.x, 
        (sy + blockSize.y - 1) / blockSize.y
    );

    init_labels<<<gridSize, blockSize>>>(out_labels, sx, sy);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; ++i) { // Iterate a fixed number of times or until convergence
        union_find<<<gridSize, blockSize>>>(labels, out_labels, sx, sy);
        cudaDeviceSynchronize();
        path_compression<<<gridSize, blockSize>>>(out_labels, sx, sy);
        cudaDeviceSynchronize();
    }
}

torch::Tensor connected_components_4(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 2, "input must be a [width, height] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t sx = input.size(0);
    const uint32_t sy = input.size(1);

    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kUInt32).device(input.device());
    torch::Tensor cc_labels = torch::zeros({H, W}, label_options);

    dim3 grid = dim3(((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS, ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cc2d::init_labeling<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), sx, sy
    );
    cudaDeviceSynchronize();
    cc2d::merge<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        sx, sy
    );
    cc2d::compression<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), sx, sy
    );
    cc2d::final_labeling<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        sx, sy
    );
    return label;
}

};

#endif