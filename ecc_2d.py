import argparse
import os
import time

import numpy as np
import torch

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # A6000, todo: Port to 8.0 for A100
from torch.utils.cpp_extension import load_inline

# ----------------- CUDA Kernel Source -----------------
cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define IMAD(a, b, c) ((a) * (b) + (c))
#define iDivUp(a, b) ((a + b - 1) / b)

// Binary search function for finding threshold bin
__device__ inline int binary_search(const float* arr, int size, float value) {
    int l = 0, r = size - 1;
    while (l <= r) {
        int m = (l + r) >> 1;
        float a = arr[m];
        if (fabs(a - value) < 1e-6f) return m;          
        if (a <  value) l = m + 1; else r = m - 1;
    }
    return (l < size) ? l : size - 1;      // safeguard (shouldn’t trigger)
}

__global__ void ECC_kernel(
    const float* __restrict__ data,
    int* __restrict__ VCEC_device,
    const float* __restrict__ ascend_unique_arr_device_,
    int imageW,
    int imageH,
    int binNum, int pitch
) {
    extern __shared__ float hist_local_[];

    // Initialize shared memory histogram and copy threshold values
    int block_pos = blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        hist_local_[block_pos] = 0;
        hist_local_[block_pos + binNum] = ascend_unique_arr_device_[block_pos];
        block_pos = block_pos + blockDim.x * blockDim.y;
    }
    __syncthreads();

    // Calculate pixel position (1-based indexing)
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x) + 1;
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) + 1;

    if (ix < imageW + 1 && iy < imageH + 1) {
        // Convert to 0-based array index
        int idx = (iy - 1) * pitch + (ix - 1);

        float change = 1;
        float c = data[idx];

        // Get neighbors with boundary handling
        // Using large positive/negative values for out-of-bounds
        float t = (iy > 1) ? data[idx - pitch] : 1e30f;           // top
        float b = (iy < imageH) ? data[idx + pitch] : -1e30f;     // bottom
        float l = (ix > 1) ? data[idx - 1] : 1e30f;                // left
        float r = (ix < imageW) ? data[idx + 1] : -1e30f;          // right
        float tl = (ix > 1 && iy > 1) ? data[idx - pitch - 1] : 1e30f;     // top-left
        float tr = (ix < imageW && iy > 1) ? data[idx - pitch + 1] : 1e30f; // top-right
        float bl = (ix > 1 && iy < imageH) ? data[idx + pitch - 1] : -1e30f; // bottom-left
        float br = (ix < imageW && iy < imageH) ? data[idx + pitch + 1] : -1e30f; // bottom-right

        bool c_l_l = (c < l);
        bool c_lq_r = (c <= r);

        if (c < t) {
            change -= 1;
            if (c_l_l && c < tl) change += 1;
            if (c_lq_r && c < tr) change += 1;
        }
        if (c <= b) {
            change -= 1;
            if (c_l_l && c <= bl) change += 1;
            if (c_lq_r && c <= br) change += 1;
        }

        change -= c_l_l ? 1.0f : 0.0f;
        change -= c_lq_r ? 1.0f : 0.0f;


        // Find bin and accumulate
        int bin = binary_search(&hist_local_[binNum], binNum, c);

        atomicAdd(&hist_local_[bin], change);
    }
    __syncthreads();

    // Write back to global memory
    block_pos = blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        atomicAdd(&VCEC_device[block_pos], (int)hist_local_[block_pos]);
        block_pos = block_pos + blockDim.x * blockDim.y;
    }
}

torch::Tensor computeECC(
    torch::Tensor data,
    torch::Tensor unique_values,
    int imageH,
    int imageW,int pitch
) {
    int binNum = unique_values.size(0);
    auto VCEC_device = torch::zeros({binNum}, torch::dtype(torch::kInt32).device(data.device()));

    
    int imageH_rounded = 512;
    if (binNum < 512) {
        imageH_rounded = 1;
        while (imageH_rounded < binNum) imageH_rounded *= 2;
        if (imageH_rounded > 512) imageH_rounded = 512;
    }

    dim3 threads;
    switch (imageH_rounded) {
        case 1: threads = dim3(512, 1); break;
        case 2: threads = dim3(256, 2); break;
        case 4: threads = dim3(128, 4); break;
        case 8: threads = dim3(64, 8); break;
        case 16: threads = dim3(32, 16); break;
        case 32: threads = dim3(16, 32); break;
        case 64: threads = dim3(16, 32); break;
        case 128: threads = dim3(16, 32); break;
        case 256: threads = dim3(16, 32); break;
        default: threads = dim3(64, 8);
    }

    if (binNum >= 1024) {
        printf("Warning: Too much shared memory used (%d bins), may fail\\n", binNum);
    }

    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    size_t shmem_size = sizeof(float) * binNum * 2;

    ECC_kernel<<<blocks, threads, shmem_size>>>(
        data.data_ptr<float>(),
        VCEC_device.data_ptr<int>(),
        unique_values.data_ptr<float>(),
        imageW,
        imageH,
        binNum, pitch
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return VCEC_device;
}
'''

cpp_source = '''
#include <torch/extension.h>

// Forward declaration
torch::Tensor computeECC(torch::Tensor data, torch::Tensor unique_values, int imageH, int imageW, int pitch);

// Python binding
torch::Tensor ecc(torch::Tensor data, torch::Tensor unique_values, int H, int W, int pitch) {
    TORCH_CHECK(data.is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(unique_values.is_cuda(), "unique_values must be a CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    TORCH_CHECK(unique_values.dtype() == torch::kFloat32, "unique_values must be float32");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(unique_values.is_contiguous(), "unique_values must be contiguous");

    return computeECC(data, unique_values, H, W, pitch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ecc", &ecc, "ECC 2D kernel");
}
'''


# ----------------- File I/O -----------------
def load_dat(path, H, W, dtype=torch.float32, device='cuda'):
    """Load .dat file and convert to PyTorch tensor"""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != H * W:
        raise ValueError(f"Expected {H * W} values, got {arr.size}")
    return torch.from_numpy(arr.reshape(H, W)).to(dtype=dtype, device=device)


def get_unique_values_sorted(data):
    """Extract unique values from data and sort them (ascending)"""
    unique_vals = torch.unique(data.flatten())
    unique_vals_sorted = torch.sort(unique_vals)[0]
    return unique_vals_sorted


# ----------------- Main -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECC 2D Implementation')
    parser.add_argument('-i', '--input', required=True, help='Input .dat file path')
    parser.add_argument('-H', '--height', type=int, required=True, help='Image height')
    parser.add_argument('-W', '--width', type=int, required=True, help='Image width')
    parser.add_argument('--output', type=str, default='ecc_2d', help='Output prefix for files')
    parser.add_argument('--max-bins', type=int, default=1024, help='Maximum number of bins')
    args = parser.parse_args()

    print("=" * 80)
    print("ECC 2D Kernel")
    print("=" * 80)

    # Compile CUDA extension
    print("Compiling CUDA extension...")
    ecc_cuda = load_inline(
        name='ecc',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
        verbose=False
    )
    print("CUDA extension compiled successfully")

    # Load data
    print(f"Loading data from {args.input}...")
    t0 = time.time()

    pitch = ((args.width + 31) // 32) * 32
    arr = np.fromfile(args.input, dtype=np.float32)
    arr2d = np.zeros((args.height, pitch), dtype=np.float32)
    arr2d[:, :args.width] = arr.reshape(args.height, args.width)
    data = torch.from_numpy(arr2d).to(device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1e3

    print(f"Data loaded: {args.height}x{args.width}")
    print(f"Load time: {dt:.2f} ms")

    # Prepare data
    H, W = data.shape
    M = H * W
    data_flat = data.reshape(-1).contiguous()

    print(f"Extracting unique values from data...")
    unique_values = get_unique_values_sorted(data)
    binNum = len(unique_values)

    print(f"Number of unique values (bins): {binNum}")
    print(f"Value range: [{unique_values[0].item():.3f}, {unique_values[-1].item():.3f}]")

    # Warm-up
    print("\nWarming up kernel...")
    for _ in range(3):
        _ = ecc_cuda.ecc(data_flat, unique_values, H, W, pitch)
    torch.cuda.synchronize()

    # Benchmark
    print("\nRunning ECC kernel...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    num_iterations = 10
    start.record()
    for _ in range(num_iterations):
        hist = ecc_cuda.ecc(data_flat, unique_values, H, W, pitch)
    end.record()
    torch.cuda.synchronize()

    t = start.elapsed_time(end) / num_iterations
    voxel_throughput = (H * W * binNum) / (t / 1000) / 1e9

    print(f"Kernel time: {t:.5f} ms (avg of {num_iterations} runs)")
    print(f"Voxel throughput: {voxel_throughput:.2f} GVox/s")
    print(f"Processing: {M:,} pixels × {binNum} bins = {M * binNum:,} operations")

    # Save results
    print(f"\nSaving results...")

    # Create combined output
    combined = np.column_stack([
        unique_values.cpu().numpy(),
        hist.cpu().numpy()
    ])
    np.savetxt(f"{args.output}_combined.csv", combined,
               delimiter=',', header='unique_value,ecc_count',
               comments='', fmt=['%.6f', '%d'])

    print(f"   Saved: {args.output}_combined.csv")

    # Statistics
    print(f"ECC Statistics:")
    print(f"Total ECC sum: {hist.sum().item()}")
    print(f"Non-zero bins: {(hist != 0).sum().item()}/{binNum}")
    print(f"Max count: {hist.max().item()} at value {unique_values[hist.argmax()].item():.3f}")
    print(f"Min count: {hist.min().item()}")

    # Sample output
    print(f"First 10 histogram values:")
    for i in range(min(10, binNum)):
        print(f"Value {unique_values[i].item():8.3f}: {hist[i].item():6d}")

    print("\nProcessing complete!")