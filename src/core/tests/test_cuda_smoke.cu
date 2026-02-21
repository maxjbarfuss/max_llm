#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    if (device_count <= 0) {
        std::cerr << "No CUDA devices detected" << '\n';
        return 1;
    }
    return 0;
}
