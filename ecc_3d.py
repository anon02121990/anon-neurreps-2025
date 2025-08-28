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

// Binary search function - finds the exact bin for a value
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

__global__ void ECC_kernel_3D(
    const float* __restrict__ data,
    int* __restrict__ VCEC_device,
    const float* __restrict__ ascend_unique_arr_device_,
    int imageW,
    int imageH,
    int imageD,
    int binNum
) {
    // Declare and initialize local histogram
    extern __shared__ float hist_local_[];
    int block_pos = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        hist_local_[block_pos] = 0;
        hist_local_[block_pos + binNum] = ascend_unique_arr_device_[block_pos];
        block_pos = block_pos + blockDim.x * blockDim.y * blockDim.z;
    }
    __syncthreads();

    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x) + 1;
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) + 1;
    const int iz = IMAD(blockDim.z, blockIdx.z, threadIdx.z) + 1;

    if (ix < imageW + 1 && iy < imageH + 1 && iz < imageD + 1) {
        float change = -1;

        // Convert to 0-based array index for center voxel
        int idx = (iz - 1) * imageW * imageH + (iy - 1) * imageW + (ix - 1);
        float ccc = data[idx];

        // Get 6 face neighbors (used 9 times each in the algorithm)
        const float OUT_OF_BOUNDS = 1e30f;
        float ccf = (iz > 1) ? data[idx - imageW * imageH] : OUT_OF_BOUNDS;      // front (z-1)
        float ccb = (iz < imageD) ? data[idx + imageW * imageH] : OUT_OF_BOUNDS;  // back (z+1)
        float ctc = (iy > 1) ? data[idx - imageW] : OUT_OF_BOUNDS;                // top (y-1)
        float cdc = (iy < imageH) ? data[idx + imageW] : OUT_OF_BOUNDS;           // down (y+1)
        float lcc = (ix > 1) ? data[idx - 1] : OUT_OF_BOUNDS;                     // left (x-1)
        float rcc = (ix < imageW) ? data[idx + 1] : OUT_OF_BOUNDS;                // right (x+1)

        // Get all 20 edge/corner neighbors for vertex checks
        // Top layer (y-1)
        float ltf = (ix > 1 && iy > 1 && iz > 1) ? 
                    data[idx - imageW * imageH - imageW - 1] : OUT_OF_BOUNDS;     // x-1, y-1, z-1
        float tcf = (iy > 1 && iz > 1) ? 
                    data[idx - imageW * imageH - imageW] : OUT_OF_BOUNDS;         // x, y-1, z-1
        float rtf = (ix < imageW && iy > 1 && iz > 1) ? 
                    data[idx - imageW * imageH - imageW + 1] : OUT_OF_BOUNDS;     // x+1, y-1, z-1
        float ltc = (ix > 1 && iy > 1) ? 
                    data[idx - imageW - 1] : OUT_OF_BOUNDS;                       // x-1, y-1, z
        float rtc = (ix < imageW && iy > 1) ? 
                    data[idx - imageW + 1] : OUT_OF_BOUNDS;                       // x+1, y-1, z
        float ltb = (ix > 1 && iy > 1 && iz < imageD) ? 
                    data[idx + imageW * imageH - imageW - 1] : OUT_OF_BOUNDS;     // x-1, y-1, z+1
        float tcb = (iy > 1 && iz < imageD) ? 
                    data[idx + imageW * imageH - imageW] : OUT_OF_BOUNDS;         // x, y-1, z+1
        float rtb = (ix < imageW && iy > 1 && iz < imageD) ? 
                    data[idx + imageW * imageH - imageW + 1] : OUT_OF_BOUNDS;     // x+1, y-1, z+1

        // Middle layer (y)
        float lcf = (ix > 1 && iz > 1) ? 
                    data[idx - imageW * imageH - 1] : OUT_OF_BOUNDS;              // x-1, y, z-1
        float rcf = (ix < imageW && iz > 1) ? 
                    data[idx - imageW * imageH + 1] : OUT_OF_BOUNDS;              // x+1, y, z-1
        float lcb = (ix > 1 && iz < imageD) ? 
                    data[idx + imageW * imageH - 1] : OUT_OF_BOUNDS;              // x-1, y, z+1
        float rcb = (ix < imageW && iz < imageD) ? 
                    data[idx + imageW * imageH + 1] : OUT_OF_BOUNDS;              // x+1, y, z+1

        // Down layer (y+1)
        float ldf = (ix > 1 && iy < imageH && iz > 1) ? 
                    data[idx - imageW * imageH + imageW - 1] : OUT_OF_BOUNDS;     // x-1, y+1, z-1
        float dcf = (iy < imageH && iz > 1) ? 
                    data[idx - imageW * imageH + imageW] : OUT_OF_BOUNDS;         // x, y+1, z-1
        float rdf = (ix < imageW && iy < imageH && iz > 1) ? 
                    data[idx - imageW * imageH + imageW + 1] : OUT_OF_BOUNDS;     // x+1, y+1, z-1
        float ldc = (ix > 1 && iy < imageH) ? 
                    data[idx + imageW - 1] : OUT_OF_BOUNDS;                       // x-1, y+1, z
        float rdc = (ix < imageW && iy < imageH) ? 
                    data[idx + imageW + 1] : OUT_OF_BOUNDS;                       // x+1, y+1, z
        float ldb = (ix > 1 && iy < imageH && iz < imageD) ? 
                    data[idx + imageW * imageH + imageW - 1] : OUT_OF_BOUNDS;     // x-1, y+1, z+1
        float dcb = (iy < imageH && iz < imageD) ? 
                    data[idx + imageW * imageH + imageW] : OUT_OF_BOUNDS;         // x, y+1, z+1
        float rdb = (ix < imageW && iy < imageH && iz < imageD) ? 
                    data[idx + imageW * imageH + imageW + 1] : OUT_OF_BOUNDS;     // x+1, y+1, z+1

        // Introduced vertices V (8) - corners of the cube
        change += (ccc < ltf && ccc < tcf && ccc < lcf && ccc < ccf && ccc < ltc && ccc < ctc && ccc < lcc);                                      // top left front vertex
        change += (ccc <= ltb && ccc <= tcb && ccc <= lcb && ccc <= ccb && ccc < ltc && ccc < ctc && ccc < lcc);                                 // top left back vertex
        change += (ccc <= rtb && ccc <= rcb && ccc <= ccb && ccc <= tcb && ccc < ctc && ccc < rtc && ccc <= rcc);                                // top right back vertex
        change += (ccc < rtf && ccc < rcf && ccc < ccf && ccc < tcf && ccc < ctc && ccc < rtc && ccc <= rcc);                                     // top right front vertex

        change += (ccc < ldf && ccc < lcf && ccc < ccf && ccc < dcf && ccc < lcc && ccc <= cdc && ccc <= ldc);                                   // down left front vertex
        change += (ccc <= ldb && ccc <= lcb && ccc <= ccb && ccc <= dcb && ccc < lcc && ccc <= cdc && ccc <= ldc);                               // down left back vertex
        change += (ccc <= rdb && ccc <= dcb && ccc <= ccb && ccc <= rcb && ccc <= rcc && ccc <= rdc && ccc <= cdc);                              // down right back vertex
        change += (ccc < rdf && ccc < dcf && ccc < ccf && ccc < rcf && ccc <= rcc && ccc <= rdc && ccc <= cdc);                                   // down right front vertex

        // Introduced edges E (12)
        change -= (ccc < tcf && ccc < ccf && ccc < ctc);       // top front edge
        change -= (ccc < ltc && ccc < ctc && ccc < lcc);       // top left edge
        change -= (ccc < ctc && ccc <= tcb && ccc <= ccb);     // top back edge
        change -= (ccc < ctc && ccc < rtc && ccc <= rcc);      // top right edge

        change -= (ccc < ccf && ccc < dcf && ccc <= cdc);      // down front edge
        change -= (ccc < lcc && ccc <= ldc && ccc <= cdc);     // down left edge
        change -= (ccc <= cdc && ccc <= ccb && ccc <= dcb);    // down back edge
        change -= (ccc <= rcc && ccc <= rdc && ccc <= cdc);    // down right edge

        change -= (ccc < lcf && ccc < ccf && ccc < lcc);       // front left edge
        change -= (ccc < lcc && ccc <= lcb && ccc <= ccb);     // back left edge
        change -= (ccc <= rcc && ccc <= ccb && ccc <= rcb);    // back right edge
        change -= (ccc < ccf && ccc < rcf && ccc <= rcc);      // front right edge

        // Introduced faces F (6)
        change += (ccc < ccf);   // front face
        change += (ccc < lcc);   // left face
        change += (ccc <= ccb);  // back face
        change += (ccc <= rcc);  // right face
        change += (ccc < ctc);   // top face
        change += (ccc <= cdc);  // down face

        atomicAdd(&hist_local_[binary_search(&hist_local_[binNum], binNum, ccc)], change);
    }
    __syncthreads();
    int t = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
    for (int i = t; i < binNum; i += blockDim.x*blockDim.y*blockDim.z) {
        atomicAdd(&VCEC_device[i], (int)hist_local_[i]);
    }
}

// Wrapper function
torch::Tensor computeECC_3D(
    torch::Tensor data,
    torch::Tensor unique_values,
    int imageH,
    int imageW,
    int imageD
) {
    int binNum = unique_values.size(0);
    auto VCEC_device = torch::zeros({binNum}, torch::dtype(torch::kInt32).device(data.device()));

    // Thread configuration for 3D
    int imageD_rounded = (binNum < 512) ? 
    (1 << (int)ceil(log2(binNum))) : 512;

    dim3 threads;
    switch (imageD_rounded) {
        case 1: threads = dim3(32, 16, 1); break;
        case 2: threads = dim3(32, 8, 2); break;  
        case 4: threads = dim3(16, 8, 4); break;
        case 8: threads = dim3(16, 4, 8); break;
        default: threads = dim3(64, 4, 2);
    }
    dim3 blocks(iDivUp(imageW, threads.x), 
                iDivUp(imageH, threads.y),
                iDivUp(imageD, threads.z));

    // Calculate shared memory size (2 * binNum floats)
    size_t shmem_size = sizeof(float) * binNum * 2;

    // Check shared memory limits
    if (shmem_size > 48 * 1024) {  // 48KB is typical limit
        printf("Error: Shared memory required (%zu bytes) exceeds limit. Reduce number of unique values.\\n", shmem_size);
        return VCEC_device;
    }

    ECC_kernel_3D<<<blocks, threads, shmem_size>>>(
        data.data_ptr<float>(),
        VCEC_device.data_ptr<int>(),
        unique_values.data_ptr<float>(),
        imageW,
        imageH,
        imageD,
        binNum
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
torch::Tensor computeECC_3D(torch::Tensor data, torch::Tensor unique_values, 
                            int imageH, int imageW, int imageD);

// Python binding
torch::Tensor ecc_3d(torch::Tensor data, torch::Tensor unique_values, 
                         int H, int W, int D) {
    TORCH_CHECK(data.is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(unique_values.is_cuda(), "unique_values must be a CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    TORCH_CHECK(unique_values.dtype() == torch::kFloat32, "unique_values must be float32");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(unique_values.is_contiguous(), "unique_values must be contiguous");

    return computeECC_3D(data, unique_values, H, W, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ecc_3d", &ecc_3d, "ECC 3D kernel");
}
'''


# ----------------- File I/O -----------------
def load_dat_3d(path, H, W, D, dtype=torch.float32, device='cuda'):
    """Load .dat file and convert to 3D PyTorch tensor"""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != H * W * D:
        raise ValueError(f"Expected {H * W * D} values, got {arr.size}")
    # Reshape to (D, H, W) - depth, height, width
    return torch.from_numpy(arr.reshape(D, H, W)).to(dtype=dtype, device=device)


def get_unique_values_sorted(data):
    """Extract unique values from data and sort them (ascending)"""
    unique_vals = torch.unique(data.flatten())
    unique_vals_sorted = torch.sort(unique_vals)[0]
    return unique_vals_sorted


# ----------------- Main -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECC 3D Implementation')
    parser.add_argument('-i', '--input', required=True, help='Input .dat file path')
    parser.add_argument('-H', '--height', type=int, required=True, help='Volume height (Y)')
    parser.add_argument('-W', '--width', type=int, required=True, help='Volume width (X)')
    parser.add_argument('-D', '--depth', type=int, required=True, help='Volume depth (Z)')
    parser.add_argument('--output', type=str, default='ecc_3d', help='Output prefix')
    parser.add_argument('--pad', action='store_true', help='Pad volume by 1 voxel on all sides')
    args = parser.parse_args()

    print("=" * 80)
    print("ECC 3D Kernel")
    print("=" * 80)

    # Compile CUDA extension
    print("Compiling CUDA extension...")
    ecc_cuda = load_inline(
        name='ecc_3d',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
        verbose=False
    )
    print("CUDA extension compiled successfully")

    # Load data
    print(f"\nLoading 3D data from {args.input}...")
    t0 = time.time()
    data = load_dat_3d(args.input, args.height, args.width, args.depth)

    # Extract ALL unique values - NO sampling or thresholding
    print(f"\nExtracting ALL unique values from 3D data...")
    unique_values = get_unique_values_sorted(data)
    binNum = len(unique_values)

    if args.pad:
        print("Padding volume by 1 voxel on all sides...")
        # Create padded array filled with a large value (to act as boundary)
        D_pad = args.depth + 2
        H_pad = args.height + 2
        W_pad = args.width + 2
        data_padded = torch.full((D_pad, H_pad, W_pad), 1e30, dtype=torch.float32, device='cuda')
        # Copy original data to center
        data_padded[1:-1, 1:-1, 1:-1] = data
        data = data_padded
        D, H, W = D_pad, H_pad, W_pad
    else:
        D, H, W = data.shape

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1e3

    print(f"Data loaded: {D}x{H}x{W} (DxHxW)")
    print(f"Load time: {dt:.2f} ms")

    # Prepare data
    M = D * H * W
    data_flat = data.reshape(-1).contiguous()

    print(f"Number of unique values (bins): {binNum}")
    print(f"Value range: [{int(unique_values[0].item())}, {int(unique_values[-1].item())}]")
    print(f"First 10 unique values: {unique_values[:10].cpu().numpy().astype(int).tolist()}")

    # Check shared memory constraints
    shmem_required = binNum * 2 * 4  # 2 floats per bin, 4 bytes per float
    shmem_limit = 48 * 1024  # 48KB typical limit

    if shmem_required > shmem_limit:
        print(f"ERROR: Too many unique values ({binNum}) for shared memory!")
        print(f"Required: {shmem_required / 1024:.1f} KB, Limit: {shmem_limit / 1024:.1f} KB")
        print(f"Consider preprocessing to reduce unique values or use a different approach.")
        exit(1)
    else:
        print(f"Shared memory OK: {shmem_required / 1024:.1f} KB / {shmem_limit / 1024:.1f} KB")

    # Warm-up
    print("\nWarming up 3D kernel...")
    for _ in range(3):
        _ = ecc_cuda.ecc_3d(data_flat, unique_values, H, W, D)
    torch.cuda.synchronize()

    # Benchmark
    print("\nRunning ECC_kernel_3D...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    num_iterations = 5  # Fewer iterations for 3D due to higher complexity
    start.record()
    for _ in range(num_iterations):
        hist_3d = ecc_cuda.ecc_3d(data_flat, unique_values, H, W, D)
    end.record()
    torch.cuda.synchronize()

    t_3d = start.elapsed_time(end) / num_iterations
    voxel_throughput = (M * binNum) / (t_3d / 1000) / 1e9

    print(f"Kernel time: {t_3d:.2f} ms (avg of {num_iterations} runs)")
    print(f"Voxel throughput: {voxel_throughput:.2f} GVox/s")
    print(f"Processing: {M:,} voxels × {binNum} bins = {M * binNum:,} operations")

    print(f"\nSaving results...")


    with open(f"{args.output}_output.txt", 'w') as f:
        unique_vals_cpu = unique_values.cpu().numpy()
        hist_cpu = hist_3d.cpu().numpy()
        for i in range(binNum):
            # Format: unique_value (as integer) followed by count
            f.write(f"{int(unique_vals_cpu[i])} {hist_cpu[i]}\n")

    print(f"Saved: {args.output}_output.txt (format: unique_value count)")

    # Also save as separate files for analysis
    np.savetxt(f"{args.output}_histogram.txt", hist_3d.cpu().numpy(), fmt='%d')
    np.savetxt(f"{args.output}_unique_values.txt", unique_values.cpu().numpy(), fmt='%.0f')

    # Create combined CSV
    combined = np.column_stack([
        unique_values.cpu().numpy(),
        hist_3d.cpu().numpy()
    ])
    np.savetxt(f"{args.output}_combined.csv", combined,
               delimiter=',', header='unique_value,ecc_count',
               comments='', fmt=['%.0f', '%d'])

    print(f"Saved: {args.output}_histogram.txt")
    print(f"Saved: {args.output}_unique_values.txt")
    print(f"Saved: {args.output}_combined.csv")

    # Statistics
    print(f"\n3D ECC Statistics:")
    print(f"Total ECC sum: {hist_3d.sum().item()}")
    print(f"Non-zero bins: {(hist_3d != 0).sum().item()}/{binNum}")


    max_idx = hist_3d.argmax()
    min_idx = hist_3d.argmin()
    print(f"Max count: {hist_3d[max_idx].item()} at value {int(unique_values[max_idx].item())}")
    print(f"Min count: {hist_3d[min_idx].item()} at value {int(unique_values[min_idx].item())}")

    print(f"\nFirst 10 ECC values (format: unique_value count):")
    for i in range(min(10, binNum)):
        print(f"   {int(unique_values[i].item())} {hist_3d[i].item()}")

    print("\n3D Processing complete!")
